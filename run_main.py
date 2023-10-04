# import os
# from subprocess import run
from abs_summarization.main import main

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == '__main__':
    # try:
    main("mt5-small", train_epoch=1, model_output="trained-mt5-ID-test")
    # except Exception:
    #     pass
    # finally:
    #     run(["shutdown", "/p"])
        