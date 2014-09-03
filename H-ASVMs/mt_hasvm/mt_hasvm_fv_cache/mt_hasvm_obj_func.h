#ifndef OBJ_FUNC_H
#define OBJ_FUNC_H

#include "mt_hasvm_model.h"
#include "mt_hasvm_fv_cache.h"
#include <string>
#include <vector>

///** -----------------------------------------------------------------
// ** Optimize the model parameters on the cache with stochastic 
// ** subgradient descent
// **/
//void sgd(double losses[3], ex_cache &E, model &M, 
//         string log_dir, string log_tag);

/** -----------------------------------------------------------------
 ** Compute the objective function value
 **/ 
void obj_val(double out[3], ex_cache &E, Moldes &M_Tree, const int model_id);

/** -----------------------------------------------------------------
 ** Compute the LSVM function value and gradient at M.w over the 
 ** cache
 **/ 
void gradient(double *obj_val, double *grad, int dim, ex_cache &E, 
              const Moldes &M_Tree, int num_threads, const int model_id);
              
/** -----------------------------------------------------------------
 ** Update various (objective function specific) bits of information 
 ** about each feature vector
 **/
void compute_info(const ex_cache &E, fv_cache &F, const Moldes &M_Tree);

#endif // OBJ_FUNC_H
