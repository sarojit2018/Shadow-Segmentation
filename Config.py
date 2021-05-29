class Config(object):

    MASK_SHAPE = (128, 128)

    BASE_PATH = './SBU-shadow/'

    BATCH_SIZE = 13 #Please adjust this based on available memory

    NUM_EPOCHS = 500

    STEPS_PER_EPOCH = 10

    VALIDATION_STEPS = 1

    TRAIN_FROM_SCRATCH = False

    ROOF_MASK_ASSIST = False

    MODEL_LR = 1e-4 #Empirical

    CVT2GRAY = False

    RGBBGR_FLIP = True

    DEBUG = False
