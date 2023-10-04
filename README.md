# t5-llm-training
An attempt to have one workflow for language models of the `t5` flavour `t5, mt5`

`master` branch fine-tunes `t5-small` currently, but can be used for other sizes of `t5` models.

`run_main.py`:
```py
main(which_llm="t5-small", train_epoch=1, model_output="trained-model-t5-small-test1")
```
will expose more parameters in the future.

`id_dataset` will fine-tune `mt5-small` with `indonesian` language trained on it, 
hopefully producing a model apt for `indonesian` article summarizations.