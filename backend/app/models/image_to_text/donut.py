"""
Donut (Document Understanding Transformer) model implementation.
Specialized for document understanding, OCR, and structured data extraction.
"""
import json
import logging
from pathlib import Path
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)

from .base import (
    BaseImageToTextModel, ImageTextResult, ImageCaptionConfig,
    ImageTaskType, ImageFormat
)
from ...base import ModelDevice

logger = logging.getLogger(__name__)


class DonutModel(BaseImageToTextModel):
    """
    Donut (Document Understanding Transformer) model wrapper.
    
    Specialized for document understanding tasks including:
    - Document parsing and information extraction
    - OCR with context understanding
    - Structured data extraction from forms, receipts, invoices
    - Document VQA
    """
    
    # Donut model variants and their specifications
    MODEL_INFO = {
        # Base models
        "naver-clova-ix/donut-base": {
            "description": "Donut base model for general document understanding",
            "max_tokens": 768,
            "image_size": 2560,
            "parameters": 165_000_000,
            "performance": "balanced",
            "memory_mb": 660,
            "tasks": ["document_understanding", "ocr", "parsing"],
            "supported_documents": ["general", "forms", "receipts", "invoices"],
        },
        "naver-clova-ix/donut-base-finetuned-cord-v2": {
            "description": "Donut fine-tuned for CORD receipt parsing",
            "max_tokens": 768,
            "image_size": 2560,
            "parameters": 165_000_000,
            "performance": "accurate",
            "memory_mb": 660,
            "tasks": ["receipt_parsing", "ocr", "structured_extraction"],
            "supported_documents": ["receipts", "CORD"],
        },
        "naver-clova-ix/donut-base-finetuned-docvqa": {
            "description": "Donut fine-tuned for Document VQA",
            "max_tokens": 768,
            "image_size": 2560,
            "parameters": 165_000_000,
            "performance": "accurate",
            "memory_mb": 660,
            "tasks": ["document_vqa", "question_answering"],
            "supported_documents": ["general", "forms", "receipts", "invoices"],
        },
        "naver-clova-ix/donut-base-finetuned-rvlcdip": {
            "description": "Donut fine-tuned for document classification (RVL-CDIP)",
            "max_tokens": 768,
            "image_size": 2560,
            "parameters": 165_000_000,
            "performance": "accurate",
            "memory_mb": 660,
            "tasks": ["document_classification"],
            "supported_documents": ["general", "RVL-CDIP"],
        },
        
        # Large models
        "naver-clova-ix/donut-large": {
            "description": "Donut large model for complex document understanding",
            "max_tokens": 1024,
            "image_size": 2560,
            "parameters": 400_000_000,
            "performance": "accurate",
            "memory_mb": 1600,
            "tasks": ["document_understanding", "ocr", "parsing", "vqa"],
            "supported_documents": ["general", "forms", "receipts", "invoices", "reports"],
        },
        
        # Specialized models
        "naver-clova-ix/donut-base-finetuned-sroie": {
            "description": "Donut fine-tuned for SROIE receipt information extraction",
            "max_tokens": 768,
            "image_size": 2560,
            "parameters": 165_000_000,
            "performance": "accurate",
            "memory_mb": 660,
            "tasks": ["receipt_parsing", "information_extraction"],
            "supported_documents": ["receipts", "SROIE"],
        },
        "naver-clova-ix/donut-base-finetuned-zuin": {
            "description": "Donut fine-tuned for ZUIN invoice parsing",
            "max_tokens": 768,
            "image_size": 2560,
            "parameters": 165_000_000,
            "performance": "accurate",
            "memory_mb": 660,
            "tasks": ["invoice_parsing", "structured_extraction"],
            "supported_documents": ["invoices", "ZUIN"],
        },
    }
    
    # Document types and their expected structures
    DOCUMENT_TYPES = {
        "receipt": {
            "fields": ["total", "date", "time", "company", "address", "items", "tax", "subtotal"],
            "format": "json",
            "prompt_template": "<s_receipt>{text}</s_receipt>",
        },
        "invoice": {
            "fields": ["invoice_number", "date", "due_date", "total", "vendor", "customer", "items", "tax"],
            "format": "json",
            "prompt_template": "<s_invoice>{text}</s_invoice>",
        },
        "form": {
            "fields": ["form_type", "fields", "values", "signatures"],
            "format": "json",
            "prompt_template": "<s_form>{text}</s_form>",
        },
        "general": {
            "fields": ["text", "structure", "entities"],
            "format": "json",
            "prompt_template": "<s_document>{text}</s_document>",
        },
    }
    
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        config: Optional[ImageCaptionConfig] = None,
        document_type: str = "general",
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Donut model.
        
        Args:
            model_name: Name of the Donut model
            config: Configuration for document understanding
            document_type: Type of document for task-specific processing
            model_kwargs: Additional arguments for model initialization
            processor_kwargs: Additional arguments for processor initialization
            **kwargs: Additional arguments passed to configuration
        """
        # Update config with Donut-specific defaults
        if config is None:
            config = ImageCaptionConfig()
        
        # Donut models typically work with larger images
        model_info = self.MODEL_INFO.get(model_name, {})
        config.image_size = (model_info.get("image_size", 2560), model_info.get("image_size", 2560))
        config.task_type = ImageTaskType.DOCUMENT_UNDERSTANDING
        
        # Increase max tokens for document understanding
        if config.generation_config.max_new_tokens < 512:
            config.generation_config.max_new_tokens = 768
        
        super().__init__(model_name, config, **kwargs)
        
        # Donut-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.processor_kwargs = processor_kwargs or {}
        self.document_type = document_type.lower()
        
        # Task-specific configuration
        self._task_config = self._get_task_config()
        
        # Generation parameters optimized for document understanding
        self._default_generation_params = {
            "max_length": self.config.generation_config.max_new_tokens,
            "min_length": self.config.generation_config.min_new_tokens,
            "temperature": 0.1,  # Lower temperature for more consistent document parsing
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.2,  # Higher penalty for document text
            "do_sample": False,  # Greedy decoding often works better for structured output
            "num_beams": 3,  # Beam search for better accuracy
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        
        # JSON extraction patterns
        self._json_patterns = [
            r'\{.*\}',  # Standard JSON object
            r'<s_.*?>.*?</s_.*?>',  # Donut XML-style tags
        ]
        
        logger.info(f"Initialized Donut model: {model_name} for {document_type} documents")
    
    def _get_task_config(self) -> Dict[str, Any]:
        """Get configuration for the specified document type."""
        default_config = {
            "prompt_template": "<s_document>{text}</s_document>",
            "output_format": "json",
            "fields": ["text", "structure"],
        }
        
        if self.document_type in self.DOCUMENT_TYPES:
            return {**default_config, **self.DOCUMENT_TYPES[self.document_type]}
        
        # Check if model name indicates a specific document type
        model_name_lower = self.model_name.lower()
        if "cord" in model_name_lower or "receipt" in model_name_lower:
            return {**default_config, **self.DOCUMENT_TYPES["receipt"]}
        elif "invoice" in model_name_lower:
            return {**default_config, **self.DOCUMENT_TYPES["invoice"]}
        elif "form" in model_name_lower:
            return {**default_config, **self.DOCUMENT_TYPES["form"]}
        elif "docvqa" in model_name_lower:
            return {
                **default_config,
                "prompt_template": "<s_docvqa><question>{question}</question><answer>{text}</answer></s_docvqa>",
                "output_format": "text",
                "fields": ["question", "answer"],
            }
        
        return default_config
    
    def load(self) -> None:
        """
        Load the Donut model and processor.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading Donut model: {self.model_name}")
            
            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    logger.info(f"Found {device_count} GPUs, using first GPU")
            else:
                self._device = "cpu"
            
            # Load processor
            logger.info(f"Loading Donut processor for {self.model_name}")
            self.processor = DonutProcessor.from_pretrained(
                self.model_name,
                **self.processor_kwargs
            )
            
            # Load model
            logger.info(f"Loading Donut model for {self.model_name}")
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                **self.model_kwargs
            )
            
            # Move model to device
            self.model.to(self._device)
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            
            # Update image size from processor
            if hasattr(self.processor, 'feature_extractor'):
                feature_extractor = self.processor.feature_extractor
                if hasattr(feature_extractor, 'size'):
                    size_dict = feature_extractor.size
                    if isinstance(size_dict, dict):
                        # Donut typically uses {"height": X, "width": Y}
                        height = size_dict.get("height", 2560)
                        width = size_dict.get("width", 1920)
                        self.image_size = (width, height)
                        logger.info(f"Updated image size from processor: {self.image_size}")
            
            logger.info(f"Successfully loaded Donut model {self.model_name} on device {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load Donut model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load Donut model {self.model_name}: {e}")
    
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        task_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image for Donut model input.
        
        Args:
            image: Input image in various formats
            task_prompt: Optional task-specific prompt
            **kwargs: Additional processing arguments
            
        Returns:
            Dictionary containing processed image tensors and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If image processing fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        try:
            # Load image
            pil_image = self._load_image(image)
            
            # Convert to RGB if necessary (Donut expects RGB)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Prepare task prompt
            if task_prompt is None:
                # Use default prompt based on document type
                if self.config.task_type == ImageTaskType.DOCUMENT_UNDERSTANDING:
                    task_prompt = self._task_config["prompt_template"].format(text="")
                elif self.config.task_type == ImageTaskType.VQA:
                    question = self.config.question or "What is in this document?"
                    task_prompt = f"<s_docvqa><question>{question}</question><answer>"
                else:
                    task_prompt = "<s_document>"
            
            # Process with Donut processor
            # Donut expects images to be processed with task prompts
            pixel_values = self.processor(
                pil_image,
                return_tensors="pt"
            ).pixel_values
            
            # Prepare decoder input
            decoder_input_ids = self.processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            
            # Move to device
            pixel_values = pixel_values.to(self._device)
            decoder_input_ids = decoder_input_ids.to(self._device)
            
            # Prepare result
            result = {
                'pixel_values': pixel_values,
                'decoder_input_ids': decoder_input_ids,
                'original_image': pil_image,
                'image_size': pil_image.size,
                'task_prompt': task_prompt,
                'model_inputs': {
                    'pixel_values': pixel_values,
                    'decoder_input_ids': decoder_input_ids,
                },
            }
            
            # Add question for VQA tasks
            if self.config.task_type == ImageTaskType.VQA and self.config.question:
                result['question'] = self.config.question
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image for Donut: {e}")
            raise RuntimeError(f"Failed to process image for Donut: {e}")
    
    def generate_text(
        self,
        image_input: Dict[str, Any],
        config: Optional[ImageCaptionConfig] = None,
        **kwargs
    ) -> ImageTextResult:
        """
        Generate text from processed document image.
        
        Args:
            image_input: Processed image input from process_image()
            config: Configuration for text generation (uses instance config if None)
            **kwargs: Additional generation arguments
            
        Returns:
            ImageTextResult containing generated text and metadata
            
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If text generation fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        if config is None:
            config = self.config
        
        try:
            import time
            start_time = time.time()
            
            # Extract model inputs
            pixel_values = image_input.get('pixel_values')
            decoder_input_ids = image_input.get('decoder_input_ids')
            
            if pixel_values is None or decoder_input_ids is None:
                raise ValueError("Processed image input missing required fields")
            
            # Prepare generation parameters
            gen_params = self._default_generation_params.copy()
            
            # Update with config
            if config.generation_config:
                gen_params.update({
                    "max_length": config.generation_config.max_new_tokens,
                    "min_length": config.generation_config.min_new_tokens,
                    "temperature": config.generation_config.temperature,
                    "top_p": config.generation_config.top_p,
                    "top_k": config.generation_config.top_k,
                    "repetition_penalty": config.generation_config.repetition_penalty,
                    "do_sample": config.generation_config.do_sample,
                    "num_beams": config.generation_config.num_beams,
                    "length_penalty": config.generation_config.length_penalty,
                    "no_repeat_ngram_size": config.generation_config.no_repeat_ngram_size,
                    "early_stopping": config.generation_config.early_stopping,
                })
            
            # Update with kwargs
            gen_params.update(kwargs)
            
            # Set seed if provided
            if config.generation_config.seed is not None:
                torch.manual_seed(config.generation_config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.generation_config.seed)
            
            # Generate document understanding output
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    **gen_params
                )
            
            # Decode generated tokens
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Post-process the generated text
            generated_text = self._post_process_output(generated_text, config.task_type)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Track performance
            self._processing_times.append(processing_time_ms)
            self._total_images_processed += 1
            
            # Parse structured output if JSON-like
            parsed_output = None
            if self._task_config.get("output_format") == "json":
                parsed_output = self._extract_json(generated_text)
            
            # Prepare result
            result = ImageTextResult(
                text=generated_text,
                model_name=self.model_name,
                model_version=self.get_version(),
                input_image_info={
                    "original_size": image_input.get('image_size'),
                    "processed_size": self.image_size,
                    "document_type": self.document_type,
                    "task_prompt": image_input.get('task_prompt'),
                },
                task_type=config.task_type,
                processing_time_ms=processing_time_ms,
                metadata={
                    "generation_params": gen_params,
                    "document_type": self.document_type,
                    "device": self._device,
                    "parsed_output": parsed_output,
                    "task_config": self._task_config,
                    "question": config.question if config.task_type == ImageTaskType.VQA else None,
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate text with Donut: {e}")
            raise RuntimeError(f"Failed to generate text with Donut: {e}")
    
    def _post_process_output(self, text: str, task_type: ImageTaskType) -> str:
        """
        Post-process Donut output for cleaner results.
        
        Args:
            text: Raw generated text
            task_type: Type of task
            
        Returns:
            Processed text
        """
        # Remove any remaining special tokens
        special_tokens = ["<pad>", "</s>", "<s>", "<unk>"]
        for token in special_tokens:
            text = text.replace(token, "")
        
        # Remove duplicate whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Task-specific processing
        if task_type == ImageTaskType.DOCUMENT_UNDERSTANDING:
            # Extract content from XML-style tags if present
            if text.startswith("<s_") and text.endswith("</s_"):
                # Find the content between tags
                match = re.search(r'<s_[^>]*>(.*?)</s_[^>]*>', text)
                if match:
                    text = match.group(1).strip()
        
        elif task_type == ImageTaskType.VQA:
            # Extract answer from VQA format
            if "<answer>" in text and "</answer>" in text:
                match = re.search(r'<answer>(.*?)</answer>', text)
                if match:
                    text = match.group(1).strip()
        
        return text
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from Donut output.
        
        Args:
            text: Generated text potentially containing JSON
            
        Returns:
            Parsed JSON dict or None if no JSON found
        """
        # Try different patterns to find JSON
        for pattern in self._json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    # Clean up the JSON string
                    json_str = json_str.strip()
                    
                    # Handle Donut's XML-style output
                    if json_str.startswith("<s_") and json_str.endswith("</s_"):
                        # Extract content and try to parse as JSON
                        content_match = re.search(r'<s_[^>]*>(.*?)</s_[^>]*>', json_str, re.DOTALL)
                        if content_match:
                            json_str = content_match.group(1).strip()
                    
                    # Parse JSON
                    return json.loads(json_str)
                    
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    try:
                        # Remove trailing commas
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        
                        # Add missing quotes around keys
                        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                        
                        return json.loads(json_str)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from Donut output: {e}")
                        continue
        
        # If no JSON found, create a simple text structure
        if text.strip():
            return {"text": text.strip()}
        
        return None
    
    def parse_document(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        document_type: Optional[str] = None,
        fields: Optional[List[str]] = None,
        return_json: bool = True,
        **kwargs
    ) -> ImageTextResult:
        """
        Parse a document and extract structured information.
        
        Args:
            image: Input document image
            document_type: Type of document (overrides instance setting)
            fields: Specific fields to extract
            return_json: Whether to return parsed JSON
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing parsed document information
        """
        # Update document type if specified
        if document_type is not None:
            self.document_type = document_type.lower()
            self._task_config = self._get_task_config()
        
        # Build task prompt based on fields
        task_prompt = None
        if fields:
            # Create a prompt requesting specific fields
            fields_str = ", ".join(fields)
            task_prompt = f"<s_{self.document_type}>Extract the following fields: {fields_str}. Output:"
        else:
            # Use default prompt
            task_prompt = self._task_config["prompt_template"].format(text="")
        
        # Update config
        config_dict = self.config.dict()
        config_dict['task_type'] = ImageTaskType.DOCUMENT_UNDERSTANDING
        config_dict.update(kwargs)
        config = ImageCaptionConfig(**config_dict)
        
        # Process image
        image_input = self.process_image(image, task_prompt=task_prompt, **kwargs)
        
        # Generate parse
        result = self.generate_text(image_input, config)
        
        # Post-process for JSON if requested
        if return_json and result.metadata.get("parsed_output"):
            # The JSON is already in metadata from generate_text
            pass
        elif return_json:
            # Try to extract JSON from text
            parsed = self._extract_json(result.text)
            if parsed:
                result.metadata["parsed_output"] = parsed
        
        return result
    
    def extract_receipt_info(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        detailed: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured information from a receipt.
        
        Args:
            image: Input receipt image
            detailed: Whether to extract detailed item information
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with receipt information
        """
        # Set up for receipt parsing
        self.document_type = "receipt"
        self._task_config = self._get_task_config()
        
        # Define fields to extract
        fields = ["total", "date", "time", "company", "address"]
        if detailed:
            fields.extend(["items", "subtotal", "tax", "payment_method"])
        
        # Parse receipt
        result = self.parse_document(
            image,
            document_type="receipt",
            fields=fields,
            return_json=True,
            **kwargs
        )
        
        # Extract parsed output
        parsed = result.metadata.get("parsed_output", {})
        
        # Ensure it has the expected structure
        if not isinstance(parsed, dict):
            parsed = {"text": result.text}
        
        # Add metadata
        parsed.update({
            "_source": "donut_receipt_parser",
            "_model": self.model_name,
            "_confidence": result.confidence,
        })
        
        return parsed
    
    def answer_document_question(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes],
        question: str,
        **kwargs
    ) -> ImageTextResult:
        """
        Answer a question about a document (Document VQA).
        
        Args:
            image: Input document image
            question: Question about the document
            **kwargs: Additional arguments
            
        Returns:
            ImageTextResult containing answer and metadata
        """
        # Set up for Document VQA
        config_dict = self.config.dict()
        config_dict['task_type'] = ImageTaskType.VQA
        config_dict['question'] = question
        config_dict.update(kwargs)
        config = ImageCaptionConfig(**config_dict)
        
        # Build VQA prompt
        task_prompt = f"<s_docvqa><question>{question}</question><answer>"
        
        # Process image
        image_input = self.process_image(image, task_prompt=task_prompt, **kwargs)
        
        # Generate answer
        return self.generate_text(image_input, config)
    
    def batch_parse_documents(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray, torch.Tensor, bytes]],
        document_types: Optional[List[str]] = None,
        batch_size: int = 2,  # Donut models are memory-intensive
        show_progress: bool = False,
        **kwargs
    ) -> List[ImageTextResult]:
        """
        Parse multiple documents in batches.
        
        Args:
            images: List of document images
            document_types: Optional list of document types
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments
            
        Returns:
            List of ImageTextResults
        """
        if document_types is not None and len(document_types) != len(images):
            raise ValueError("Number of document_types must match number of images")
        
        results = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_types = document_types[i:i + batch_size] if document_types else None
            
            if show_progress:
                batch_num = i // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for j, image in enumerate(batch_images):
                doc_type = batch_types[j] if batch_types else None
                
                try:
                    if doc_type:
                        result = self.parse_document(image, document_type=doc_type, **kwargs)
                    else:
                        result = self.parse_document(image, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to parse document {i + j}: {e}")
                    # Create error result
                    error_result = ImageTextResult(
                        text="[ERROR] Failed to parse document",
                        model_name=self.model_name,
                        model_version=self.get_version(),
                        input_image_info={"error": str(e)},
                        task_type=ImageTaskType.DOCUMENT_UNDERSTANDING,
                        metadata={"error": True, "error_message": str(e)}
                    )
                    results.append(error_result)
        
        return results
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the Donut model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "document_types": model_info.get("supported_documents", ["general"]),
            "tasks": model_info.get("tasks", []),
            "max_image_size": self.image_size,
            "max_tokens": model_info.get("max_tokens", 768),
            "supports_json_output": True,
            "supports_document_vqa": "document_vqa" in model_info.get("tasks", []),
            "supports_structured_extraction": any(
                t in ["parsing", "structured_extraction", "information_extraction"] 
                for t in model_info.get("tasks", [])
            ),
        }
        
        return capabilities
    
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        if not self._is_loaded:
            return "unknown"
        
        try:
            # Try to get version from config
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, '_commit_hash'):
                    return config._commit_hash[:8]
                elif hasattr(config, 'model_type'):
                    return f"donut-{config.model_type}"
            
            return "1.0.0"
            
        except:
            return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded Donut model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add Donut-specific information
        donut_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        info.update({
            "document_type": self.document_type,
            "task_config": self._task_config,
            "model_capabilities": self.get_model_capabilities(),
            "model_specs": donut_info,
        })
        
        return info
    
    def optimize_for_inference(
        self,
        use_half_precision: bool = True,
        compile_model: bool = False,
        **kwargs
    ) -> None:
        """
        Optimize Donut model for inference performance.
        
        Args:
            use_half_precision: Whether to use half precision (FP16)
            compile_model: Whether to compile model with torch.compile
            **kwargs: Additional optimization arguments
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        logger.info(f"Optimizing Donut model {self.model_name} for inference")
        
        try:
            # Apply half precision
            if use_half_precision and self._device == "cuda":
                self.model = self.model.half()
                logger.info("Applied half precision (FP16)")
            
            # Compile model with torch.compile
            if compile_model:
                try:
                    import torch
                    if hasattr(torch, 'compile'):
                        self.model = torch.compile(self.model)
                        logger.info("Applied torch.compile optimization")
                    else:
                        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}")
        
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            raise RuntimeError(f"Failed to optimize model: {e}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available Donut models.
        
        Returns:
            Dictionary of model names to model information
        """
        return cls.MODEL_INFO.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific Donut model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        return cls.MODEL_INFO.get(model_name)
    
    def __repr__(self) -> str:
        """String representation."""
        loaded = self._is_loaded
        device = self._device if loaded else "not loaded"
        doc_type = self.document_type
        return (
            f"DonutModel(model_name={self.model_name}, "
            f"document_type={doc_type}, loaded={loaded}, device={device})"
        )


# Register with factory
try:
    from .base import ImageToTextModelFactory
    ImageToTextModelFactory.register_model('donut', DonutModel)
    logger.info("Registered DonutModel with ImageToTextModelFactory")
except ImportError:
    logger.warning("Could not register DonutModel with factory")


__all__ = [
    'DonutModel',
    'MODEL_INFO',
    'DOCUMENT_TYPES',
]