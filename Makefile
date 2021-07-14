MAIN = scripts/main.py
URL = https://body-composition.s3.amazonaws.com

CUDA_VISIBLE_DEVICES ?=
TF_CPP_MIN_LOG_LEVEL ?= 3
PY ?= \
    CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
    TF_CPP_MIN_LOG_LEVEL=$(TF_CPP_MIN_LOG_LEVEL) \
    python

PYTHONPATH := .:$(PYTHONPATH)
CURL = curl
TAR = tar -zx -C .

ROOT_DIR ?= .
DO ?= predict
STAGE ?=
CHECKPOINT_DIR ?= models/$(STAGE)
RESULT_DIR ?= results/$(STAGE)

ifeq ($(STAGE),first)
    GIN = configs/first-stage.gin
    PREDICT_PATH = data/studies/1014.nii.gz
    PREDICT_BATCH_SIZE = 64
    PREDICT_ARGS = \
        --gin_param "FirstStage.z_step = 0.25" \
        --gin_param "FirstStage.t_dim = 0.01" \
        --gin_param "FirstStage.t_dim_max = 2.0" \
        --gin_param "FirstStage.z_dim_interp = 0.1" \
        --gin_param "FirstStage.z_dim_filter = 25.0" \
        --gin_param "FirstStage.prob_sample = 10000" \
        --gin_param "FirstStage.hist_thresh = 0.1" \
        --gin_param "FirstStage.sigma = 1" \
        --gin_param "FirstStage.topk = 5"

else ifeq ($(STAGE),second)
    GIN = configs/second-stage.gin
    PREDICT_PATH = results/first/*.nii.gz
    PREDICT_BATCH_SIZE = 16

endif

# building arguments

ARGS += \
    --do $(DO) \
    --root_dir $(ROOT_DIR)

ifeq ($(DO),train_eval)
    ARGS += \
        --gin_file $(GIN)

else ifeq ($(DO),predict)
    ARGS += \
        --predict_path "$(ROOT_DIR)/$(PREDICT_PATH)" \
        --checkpoint_dir "$(ROOT_DIR)/$(CHECKPOINT_DIR)"

    ifneq ($(RESULT_DIR),)
        ARGS += \
            --result_dir "$(ROOT_DIR)/$(RESULT_DIR)"
    endif

    ARGS += \
        $(PREDICT_ARGS) \
        --gin_param "PredictSpec.batch_size = $(PREDICT_BATCH_SIZE)"
endif

main:
	$(PY) $(MAIN) $(ARGS)

pull-model:
	$(CURL) $(URL)/models.tar.gz | $(TAR)

pull-data:
	$(CURL) $(URL)/data.tar.gz | $(TAR)
