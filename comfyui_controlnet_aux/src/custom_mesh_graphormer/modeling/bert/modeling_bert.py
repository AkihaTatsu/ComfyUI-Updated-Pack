from transformers.models.bert import modeling_bert

for symbol in dir(modeling_bert):
    if not symbol.startswith("_"):
        globals()[symbol] = getattr(modeling_bert, symbol)

# Handle transformers 5.0.0+ compatibility
# load_tf_weights_in_bert was removed in transformers 5.0.0
# Provide a stub function for compatibility with older code that imports it
if 'load_tf_weights_in_bert' not in globals():
    def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
        """
        Stub function for compatibility with transformers 5.0.0+
        This function was removed in transformers 5.0.0 as TensorFlow weight loading
        is no longer supported in the same way.
        """
        raise NotImplementedError(
            "load_tf_weights_in_bert is not available in transformers 5.0.0+. "
            "Please use PyTorch weights instead."
        )
