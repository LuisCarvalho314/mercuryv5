PYTHON ?= python3
STUDY_CONFIG ?= configs/studies/default.json
STUDY_ROOT ?= results/current/studies
STUDY_NAME ?= default
MAX_WORKERS ?= 1
FLAGS ?=
WANDB_PROJECT ?= mercuryv5
WANDB_MODE ?= online
WANDB_ENTITY ?=
WANDB_GROUP ?=
WANDB_TAGS ?=
WANDB_JOB_TYPE ?= study
WANDB_LOG_ARTIFACTS ?= True
POCML_EPOCHS ?= 8
POCML_STATE_DIM ?= 64
POCML_RF_DIM ?= 512

.PHONY: help study study-resume study-retry study-retry-incomplete study-wandb study-wandb-offline study-compare-pocml study-compare-cscg study-compare-all

help:
	@echo "Targets:"
	@echo "  make study         # run study grid from STUDY_CONFIG"
	@echo "  make study-resume  # skip already completed runs"
	@echo "  make study-retry   # retry failed runs from study_errors.jsonl"
	@echo "  make study-retry-incomplete # retry runs missing outputs from manifest"
	@echo "  make study-wandb   # run study with W&B logging"
	@echo "  make study-wandb-offline # run study with W&B offline mode"
	@echo "  make study-compare-pocml # run study and train POCML baseline per run"
	@echo "  make study-compare-cscg # run study and train CSCG baseline per run"
	@echo "  make study-compare-all # run study with Mercury + POCML + CSCG"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON, STUDY_CONFIG, STUDY_ROOT, STUDY_NAME, MAX_WORKERS, FLAGS"
	@echo "  WANDB_PROJECT, WANDB_MODE, WANDB_ENTITY, WANDB_GROUP, WANDB_TAGS,"
	@echo "  WANDB_JOB_TYPE, WANDB_LOG_ARTIFACTS"
	@echo "  POCML_EPOCHS, POCML_STATE_DIM, POCML_RF_DIM"

study:
	@echo "[study] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		$(FLAGS)

study-resume:
	@echo "[study-resume] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study --resume \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		$(FLAGS)

study-retry:
	@echo "[study-retry] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study --retry_failed \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		$(FLAGS)

study-retry-incomplete:
	@echo "[study-retry-incomplete] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study --retry_incomplete \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		$(FLAGS)

study-wandb:
	@echo "[study-wandb] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		--wandb True \
		--wandb_project $(WANDB_PROJECT) \
		--wandb_mode $(WANDB_MODE) \
		--wandb_entity $(WANDB_ENTITY) \
		--wandb_group $(WANDB_GROUP) \
		--wandb_tags $(WANDB_TAGS) \
		--wandb_job_type $(WANDB_JOB_TYPE) \
		--wandb_log_artifacts $(WANDB_LOG_ARTIFACTS) \
		$(FLAGS)

study-wandb-offline:
	@echo "[study-wandb-offline] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		--wandb True \
		--wandb_project $(WANDB_PROJECT) \
		--wandb_mode offline \
		--wandb_entity $(WANDB_ENTITY) \
		--wandb_group $(WANDB_GROUP) \
		--wandb_tags $(WANDB_TAGS) \
		--wandb_job_type $(WANDB_JOB_TYPE) \
		--wandb_log_artifacts $(WANDB_LOG_ARTIFACTS) \
		$(FLAGS)

study-compare-pocml:
	@echo "[study-compare-pocml] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		--baseline_pocml True \
		--pocml_epochs $(POCML_EPOCHS) \
		--pocml_state_dim $(POCML_STATE_DIM) \
		--pocml_random_feature_dim $(POCML_RF_DIM) \
		$(FLAGS)

study-compare-cscg:
	@echo "[study-compare-cscg] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		--baseline_cscg True \
		$(FLAGS)

study-compare-all:
	@echo "[study-compare-all] STUDY_NAME=$(STUDY_NAME) STUDY_CONFIG=$(STUDY_CONFIG) MAX_WORKERS=$(MAX_WORKERS)"
	$(PYTHON) main.py --study \
		--study_config $(STUDY_CONFIG) \
		--study_root $(STUDY_ROOT) \
		--study_name $(STUDY_NAME) \
		--max_workers $(MAX_WORKERS) \
		--baseline_all True \
		--pocml_epochs $(POCML_EPOCHS) \
		--pocml_state_dim $(POCML_STATE_DIM) \
		--pocml_random_feature_dim $(POCML_RF_DIM) \
		$(FLAGS)
