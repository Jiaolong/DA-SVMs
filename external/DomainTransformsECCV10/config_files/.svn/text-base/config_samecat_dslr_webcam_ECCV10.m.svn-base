% Settings used for the ECCV10 paper, Table 1
function PARAM = config_samecat_dslr_webcam_ECCV10(datadir)

PARAM.NUM_RUNS = 5;
PARAM.image_dirs = {'dslr/images/' 'webcam/images/'};
PARAM.domains = {'dslr/interest_points/' 'webcam/interest_points/' };
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
PARAM.gamma_set = 10^2;  % can be a range, e.g. 10.^(-2:4);

% hold out categories for testing?
PARAM.testOnNewCategories = 0;

% which categories to hold out
PARAM.numc_train = 15;
PARAM.classes_train = 1:PARAM.numc_train;
PARAM.classes_test = PARAM.numc_train+1:length(PARAM.categories);

%no. training images
PARAM.num_training_A = 8;  
PARAM.num_training_B = 3;

%the type of constraints to use: corresp, interdomain, allpairs
PARAM.symm_constraint_type = 'interdomain';
PARAM.asymm_constraint_type = 'interdomain';

% which object IDs to train and test on
PARAM.trainIDs_A = [1 2 3];
PARAM.testIDs_A = [4 5];
PARAM.trainIDs_B = [1]; % webcam
PARAM.testIDs_B = [4 5];

PARAM.validationIDs_A = []; %[2 3];
PARAM.validationIDs_B = []; %[4 5];

PARAM.pca_dim = 500;

end