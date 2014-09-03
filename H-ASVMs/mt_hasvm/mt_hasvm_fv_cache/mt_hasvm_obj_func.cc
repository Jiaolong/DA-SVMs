#include "mt_hasvm_obj_func.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace std;

/** -----------------------------------------------------------------
 ** Softmax parameters
 **  softmax(x_1,...,x_i) = 1/beta * log[sum_i[exp(beta*x_i)]]
 **/
static const double beta = 1000.0;
static const double inv_beta = 1.0 / beta;

// Indexes for the objective function value on background examples, 
// foreground examples, and the regularization term
enum { OBJ_VAL_BG = 0, OBJ_VAL_FG, OBJ_VAL_RG, OBJ_VAL_LEN };

/** -----------------------------------------------------------------
 ** Compute the value of the object function on the cache
 **/
void obj_val(double out[OBJ_VAL_LEN], ex_cache &E, Moldes &M_Tree, const int model_id) {
    // TODO: consider merging with gradient()
    model M    = M_Tree[model_id];
    model M_pa = M_Tree[M.parent_id];
    double **w = M.w;
    double **w_pa = M_pa.w;

    out[OBJ_VAL_BG] = 0.0; // background examples (from neg)
    out[OBJ_VAL_FG] = 0.0; // foreground examples (from pos)
    out[OBJ_VAL_RG] = 0.0; // regularization

    if (M.reg_type == model::REG_L2) {
        // compute ||w||^2
        for (int b = 0; b < M.num_blocks; b++) {
            const double *wb = w[b];
            const double *wb_pa = w_pa[b];
            double reg_mult  = M.reg_mult[b];
            for (int k = 0; k < M.block_sizes[b]; k++)
                out[OBJ_VAL_RG] += (wb[k]-wb_pa[k]) * (wb[k]-wb_pa[k]) * reg_mult;
        }
        out[OBJ_VAL_RG] *= 0.5;
    } else if (M.reg_type == model::REG_MAX) {
        // Compute softmax regularization cost
        double hnrms2[M.num_components];
        double max_hnrm2 = -INFINITY;
        for (int c = 0; c < M.num_components; c++) {
            if (M.component_sizes[c] == 0)
                continue;

            double val = 0;
            for (int i = 0; i < M.component_sizes[c]; i++) {
                int b            = M.component_blocks[c][i];
                double reg_mult  = M.reg_mult[b];
                double *wb       = w[b];
                double *wb_pa    = w_pa[b];
                double block_val = 0;
                for (int k = 0; k < M.block_sizes[b]; k++)
                    block_val += (wb[k]-wb_pa[k]) * (wb[k]-wb_pa[k]) * reg_mult;
                val += block_val;
            }
            // val = 1/2 ||w_c||^2
            val = 0.5 * val;
            hnrms2[c] = val;
            if (val > max_hnrm2)
                max_hnrm2 = val;
        }

        double Z = 0;
        for (int c = 0; c < M.num_components; c++) {
            if (M.component_sizes[c] == 0)
                continue;

            double a = exp(beta * (hnrms2[c] - max_hnrm2));
            Z += a;
        }

        out[OBJ_VAL_RG] = max_hnrm2 + inv_beta * log(Z);
    }

    for (ex_iter i = E.begin(), i_end = E.end(); i != i_end; ++i) {
        fv_iter I = i->begin;
        fv_iter belief_I = i->begin;
        double V = -INFINITY;
        double belief_score = 0;
        int subset = OBJ_VAL_FG;

        // Check if the current example can be assigned to this model.
        // The example's model id is coressponding to the leaf node id in the model tree,
        // so we need to check if the current model is also at the parent node.
        int fv_model_id = I->key[fv::KEY_MODEL_ID];
        bool skip = true;
        if(fv_model_id == model_id)
            skip = false;
        else { // Look for the parent
            int fv_cur_id = fv_model_id;
            do{
                model M_fv = M_Tree[fv_cur_id];
                int fv_parent_id = M_fv.parent_id;
                if(fv_parent_id !=0 && fv_parent_id == model_id){
                    skip = false;
                    break;
                }
                fv_cur_id = fv_parent_id;
            }while(fv_cur_id!=0);
        }
        if(skip)
            continue;

        for (fv_iter m = i->begin; m != i->end; ++m) {
            double score = M.score_fv(*m);

            if (m->is_belief) {
                belief_score = score;
                belief_I = m;
                if (m->is_zero)
                    subset = OBJ_VAL_BG;
            }

            score += m->loss;
            if (score > V) {
                I = m;
                V = score;
            }
        }
        out[subset] += M.C * (V - belief_score);
    }
}

/** -----------------------------------------------------------------
 ** Compute score and margin for each feature vector.
 */
void compute_info(const ex_cache &E, fv_cache &F, const Moldes &M_Tree) {
    const int num_examples = E.size();

    //#pragma omp parallel for schedule(static)
    for (int q = 0; q < num_examples; q++) {
        ex i = E[q];
        double belief_score = 0;
        for (fv_iter m = i.begin; m != i.end; ++m) {
            int model_id = m->key[fv::KEY_MODEL_ID];
            double score = M_Tree[model_id].score_fv(*m);

            // record score of belief
            if (m->is_belief)
                belief_score = score;

            m->score = score;
        }

        // compute margin for each entry in this example
        for (fv_iter m = i.begin; m != i.end; ++m) {
            m->margin = belief_score - (m->score + m->loss);
            mexPrintf("Id: %d, loss: %f, y: %d, belief score: %f, margin: %f\n", 
            m->key[fv::KEY_DATA_ID], m->loss, m->key[fv::KEY_Y], belief_score, m->margin);
        }
    }
}


/** -----------------------------------------------------------------
 ** Update the gradient by adding to it the subgradient from one
 ** example.
 */
static inline void update_gradient(const model &M, const fv_iter I, 
                                   double **grad_blocks, double mult) {
    // short circuit if the feat vector is zero
    if (I->is_zero)
        return;

    const float *feat = I->feat;
    int nbls          = I->num_blocks;
    const int *bls    = I->block_labels;

    for (int j = 0; j < nbls; j++) {
        int b             = bls[j];
        double *ptr_grad  = grad_blocks[b];
        if (M.learn_mult[b] != 0)
            for (int k = 0; k < M.block_sizes[b]; k++)
                *(ptr_grad++) += mult * feat[k];
        feat += M.block_sizes[b];
    }
}

/** -----------------------------------------------------------------
 ** Compute the gradient and value of the objective function at the
 ** point M.w.
 */
void gradient(double *obj_val_out, double *grad, const int dim, 
              ex_cache &E, const Moldes &M_Tree, int num_threads, const int model_id) {

    model M    = M_Tree[model_id];
    model M_pa = M_Tree[M.parent_id];

    // Gradient per thread
    double **grad_threads = new (nothrow) double*[num_threads];
    check(grad_threads != NULL);

    // Pointer to the start of each block in each per-thread gradient
    double ***grad_blocks = new (nothrow) double**[num_threads];
    check(grad_blocks != NULL);

    // Objective function value per thread
    double *obj_vals = new (nothrow) double[num_threads];
    int    *num_validexamples = new (nothrow) int[num_threads];
    check(obj_vals != NULL);
    check(num_validexamples != NULL);
    fill(obj_vals, obj_vals+num_threads, 0);
    fill(num_validexamples, num_validexamples+num_threads, 0);

#pragma omp parallel shared(grad_threads, grad_blocks, obj_vals)
    {
        double *grad_th = new (nothrow) double[dim];
        check(grad_th != NULL);
        fill(grad_th, grad_th+dim, 0);

        double **grad_blocks_th = new (nothrow) double*[M.num_blocks];
        check(grad_blocks_th != NULL);
        int off = 0;
        for (int i = 0; i < M.num_blocks; i++) {
            grad_blocks_th[i] = grad_th + off;
            off += M.block_sizes[i];
        }

        const int th_id = omp_get_thread_num();
        grad_threads[th_id] = grad_th;
        grad_blocks[th_id]  = grad_blocks_th;

        const int num_examples = E.size();

#pragma omp for schedule(static)
        for (int q = 0; q < num_examples; q++) {
            // Check margin-bound pruning condition
            // See Appendix B of my dissertation for details
//            E[q].hist++;
//            int hist = E[q].hist;
//            if (hist < model::hist_size) {
//                double skip = E[q].margin_bound
//                        - M.dw_norm_hist[hist]
//                        * (E[q].belief_norm + E[q].max_nonbelief_norm);
//                if (skip > 0)
//                    continue;
//            }

            ex i = E[q];

            fv_iter I = i.begin;
            fv_iter belief_I = i.begin;
            double V = -INFINITY;
            double belief_score = 0;
            double max_nonbelief_score = -INFINITY;

            // Check if the current example can be assigned to this model.
            // The example's model id is coressponding to the leaf node id in the model tree,
            // so we need to check if the current model is also at the parent node.
            int fv_model_id = I->key[fv::KEY_MODEL_ID];
            bool skip = true;
            if(fv_model_id == model_id)
                skip = false;
            else { // Look for the parent
                int fv_cur_id = fv_model_id;
                do{
                    model M_fv = M_Tree[fv_cur_id];
                    int fv_parent_id = M_fv.parent_id;
                    if(fv_parent_id !=0 && fv_parent_id == model_id) {
                        skip = false;
                        break;
                    }
                    fv_cur_id = fv_parent_id;
                }while(fv_cur_id!=0);
            }
            if(skip)
                continue;

            for (fv_iter m = i.begin; m != i.end; ++m) {
                double score = M.score_fv(*m);
                double loss_adj_score = score + m->loss;

                // record score of belief
                if (m->is_belief) {
                    belief_score = score;
                    belief_I = m;
                } else if (loss_adj_score > max_nonbelief_score) {
                    max_nonbelief_score = loss_adj_score;
                }

                if (loss_adj_score > V) {
                    I = m;
                    V = loss_adj_score;
                }
            }

            num_validexamples[th_id] += 1;
            obj_vals[th_id] += M.C * (V - belief_score);
            E[q].margin_bound = belief_score - max_nonbelief_score;
            E[q].hist = 0;

            if (I != belief_I) {
                update_gradient(M, I, grad_blocks_th, M.C);
                update_gradient(M, belief_I, grad_blocks_th, -1.0 * M.C);
            }
        }
    }

    int count = 0;
    for (int t = 0; t < num_threads; t++)
        count += num_validexamples[t];
    // mexPrintf("Model id: %d, num_examples: %d\n", model_id, count);

    double obj_val = -INFINITY;

    double **w = M.w;
    double **w_pa = M_pa.w;
    if (M.reg_type == model::REG_L2) {
        // Cost and gradient of the L2 regularization term
        obj_val = 0;

        for (int b = 0; b < M.num_blocks; b++) {
            const double *wb = w[b];
            const double *wb_pa = w_pa[b];
            double reg_mult  = M.reg_mult[b];
            double *ptr_grad = grad_blocks[0][b];
            for (int k = 0; k < M.block_sizes[b]; k++) {
                *(ptr_grad++) += (wb[k]-wb_pa[k]) * reg_mult;
                obj_val += (wb[k]-wb_pa[k]) * (wb[k]-wb_pa[k]) * reg_mult;
            }
        }
        obj_val *= 0.5;
    } else if (M.reg_type == model::REG_MAX) {
        // Cost and gradient of the softmax regularization term
        double hnrms2[M.num_components];
        double max_hnrm2 = -INFINITY;
        for (int c = 0; c < M.num_components; c++) {
            if (M.component_sizes[c] == 0)
                continue;

            double val = 0;
            for (int i = 0; i < M.component_sizes[c]; i++) {
                int b            = M.component_blocks[c][i];
                double reg_mult  = M.reg_mult[b];
                double *wb       = w[b];
                double *wb_pa    = w_pa[b];
                double block_val = 0;
                for (int k = 0; k < M.block_sizes[b]; k++)
                    block_val += (wb[k]-wb_pa[k]) * (wb[k]-wb_pa[k]) * reg_mult;
                val += block_val;
            }
            // val = 1/2 ||w_c||^2
            val = 0.5 * val;
            hnrms2[c] = val;
            if (val > max_hnrm2)
                max_hnrm2 = val;
        }

        double pc[M.num_components];
        double Z = 0;
        for (int c = 0; c < M.num_components; c++) {
            if (M.component_sizes[c] == 0)
                continue;

            double a = exp(beta * (hnrms2[c] - max_hnrm2));
            pc[c] = a;
            Z += a;
        }
        double inv_Z = 1.0 / Z;

        obj_val = max_hnrm2 + inv_beta * log(Z);

        for (int c = 0; c < M.num_components; c++) {
            if (M.component_sizes[c] == 0)
                continue;

            double cmult = pc[c] * inv_Z;
            for (int i = 0; i < M.component_sizes[c]; i++) {
                int b = M.component_blocks[c][i];
                double reg_mult = M.reg_mult[b];
                double *wb = w[b];
                double *wb_pa = w_pa[b];
                double *ptr_grad = grad_blocks[0][b];
                for (int k = 0; k < M.block_sizes[b]; k++)
                    *(ptr_grad++) += (wb[k]- wb_pa[k]) * reg_mult * cmult;
            }
        }
    }

    for (int t = 0; t < num_threads; t++) {
        obj_val += obj_vals[t];
        double *ptr_grad = grad_threads[t];
        for (int i = 0; i < dim; i++)
            grad[i] += ptr_grad[i];
        delete [] grad_threads[t];
        delete [] grad_blocks[t];
    }
    delete [] grad_threads;
    delete [] grad_blocks;
    delete [] obj_vals;

    *obj_val_out = obj_val;
}


