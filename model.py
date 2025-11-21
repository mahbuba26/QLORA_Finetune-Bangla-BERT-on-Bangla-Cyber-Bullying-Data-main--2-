"""
Generic Transformer-based Multi-Label Classifier
Supports any transformer model (BERT, RoBERTa, XLM-RoBERTa, etc.) through AutoModel
"""

import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig


class TransformerMultiLabelClassifier(nn.Module):
    """
    Generic transformer-based multi-label classifier.
    Works with any transformer model from HuggingFace (BERT, RoBERTa, XLM-RoBERTa, etc.)
    """
    
    def __init__(self, model_name, num_labels, dropout=0.1,use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05,use_quantization=False, quant_type='4bit'):
        """
        Initialize the multi-label classifier.
        
        Args:
            model_name (str): Name or path of pre-trained transformer model
            num_labels (int): Number of labels for multi-label classification
            dropout (float): Dropout rate for regularization
            use_lora (bool): Whether to apply LoRA to the model
            lora_r (int): LoRA rank (smaller = fewer parameters)
            lora_alpha (int): LoRA scaling factor
            lora_dropout (float): LoRA dropout rate
        """
        super(TransformerMultiLabelClassifier, self).__init__()
            # --- ADD QUANTIZATION CONFIG ---
        bnb_config = None
        device_map = None # Let train.py handle .to(device) unless quantizing
        
        if use_quantization and torch.cuda.is_available():
            print(f"Applying {quant_type} quantization...")
            if quant_type == '4bit':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif quant_type == '8bit':
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # device_map="auto" is required for bitsandbytes quantization
            # It automatically places the quantized model on the GPU
            device_map = "auto"
        # --- END OF ADDITION
        
        # Auto-detect and load any transformer model
        self.encoder = AutoModel.from_pretrained(model_name, quantization_config=bnb_config, # <--- PASS BITSANDBYTES CONFIG
            device_map=device_map)            # <--- PASS DEVICE MAP)
        
        # Get the hidden size from the model's config
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size

        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # Sequence classification task
                r=lora_r,  # Rank of LoRA matrices
                lora_alpha=lora_alpha,  # Scaling factor
                lora_dropout=lora_dropout,  # Dropout for LoRA layers
                bias="none",  # Can be "none", "all", or "lora_only"
                target_modules=["query", "value"],  # Apply LoRA to attention layers
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()  # See trainable params
        
        # Classification head with intermediate layer for better feature extraction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            labels: Ground truth labels (optional, for loss calculation)
            
        Returns:
            dict: Dictionary containing loss (if labels provided) and logits
        """
        # Get encoder outputs (works for BERT, RoBERTa, XLM-RoBERTa, etc.)
        #outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Extract CLS token representation (first token)
        # This pattern works for most transformer models
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(cls_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}
    
    def freeze_base_layers(self):
        """
        Freeze encoder parameters for feature extraction.
        This prevents updating the pre-trained weights during training.
        
        Note: This is now an instance method (bug fix from original code)
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Count frozen parameters for logging
        frozen_params = sum(p.numel() for p in self.encoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters")
        print(f"Trainable parameters: {total_params - frozen_params:,}")
    
    def unfreeze_base_layers(self):
        """
        Unfreeze encoder parameters (useful for fine-tuning after feature extraction).
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        
     
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"All parameters unfrozen. Trainable parameters: {trainable_params:,}")