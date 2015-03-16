function PARAM = config_diffcat_webcam_dslr(datadir)

PARAM.NUM_RUNS = 5;
PARAM.image_dirs = {'webcam/images/' 'dslr/images/'};
PARAM.domains = {'webcam/interest_points/' 'dslr/interest_points/'};
PARAM.base_dir = datadir;
PARAM.histfile =   {'histogram_*.SURF_SURF.amazon_800.SURF_SURF.mat', ...
	      'histogram_*.SURF_SURF.amazon_800.SURF_SURF.mat'};
% we have 31 categories
PARAM.categories = {  'back_pack'    'bike'    'bike_helmet'    'bookcase'    'bottle'    'calculator'    'desk_chair'    'desk_lamp'    'desktop_computer'    'file_cabinet'    'headphones'    'keyboard'    'laptop_computer'    'letter_tray'    'mobile_phone'    'monitor'    'mouse'    'mug'    'paper_notebook'    'pen'    'phone'    'printer'    'projector'    'punchers'    'ring_binder'    'ruler'    'scissors'    'speaker'    'stapler'    'tape_dispenser'    'trash_can' };

%linear or nonlinear?
PARAM.use_Gaussian_kernel = 1;

%number of NNs to use for the kNN classifier
PARAM.k = 1;

% gamma values to try
PARAM.gamma_set = 10^0;  % can be a range, e.g. 10.^(-2:4);

% hold out categories for testing?
PARAM.testOnNewCategories = 1;

% which categories to hold out
PARAM.numc_train = 15;
PARAM.classes_train = 1:PARAM.numc_train;
PARAM.classes_test = PARAM.numc_train+1:length(PARAM.categories);

%no. training images FOR TRAINING TRANSFORM (on training classes)
PARAM.num_training_A = inf;  
PARAM.num_training_B = inf;  % use all

%no. test images for testing transform:
PARAM.num_testing_A = 20;  % no. labeled images in A
PARAM.num_testing_B = 10;  % no. test images in B

%the type of constraints to use: corresp, interdomain, allpairs
PARAM.symm_constraint_type = 'corresp'; %'interdomain';
PARAM.asymm_constraint_type = 'corresp'; %'interdomain';

% which object IDs to train and test on
PARAM.trainIDs_A = [1 2 3 4 5];
PARAM.trainIDs_B = [1 2 3 4 5]; 
PARAM.testIDs_A = [1 2 3 4 5];  % training for knn
PARAM.testIDs_B = [1 2 3 4 5];

PARAM.validationIDs_A = []; %[2 3];
PARAM.validationIDs_B = []; %[4 5];

PARAM.pca_dim = 500;

end