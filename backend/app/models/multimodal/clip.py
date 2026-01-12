"""
CLIP (Contrastive Language-Image Pre-training) model implementation.

CLIP is a multimodal model that learns visual concepts from natural language 
supervision. It can be used for image-text matching, zero-shot classification,
and multimodal embeddings.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base import (
    BaseMultimodalModel, MultimodalInput, MultimodalResult,
    MultimodalTaskType, ImageInput, TextInput, ModelDevice
)

logger = logging.getLogger(__name__)


class CLIPModelWrapper(BaseMultimodalModel):
    """
    CLIP model wrapper for WorldBrief360.
    
    CLIP (Contrastive Language-Image Pre-training) is a multimodal model
    that connects images and text through a shared embedding space.
    
    Capabilities:
    - Image-text similarity matching
    - Zero-shot image classification
    - Multimodal embeddings
    - Text-based image retrieval
    - Image-based text retrieval
    """
    
    # CLIP model variants and their specifications
    MODEL_INFO = {
        "openai/clip-vit-base-patch32": {
            "description": "CLIP base model with ViT-B/32 image encoder",
            "image_size": (224, 224),
            "embedding_dim": 512,
            "parameters": 151_000_000,
            "performance": "good",
            "memory_mb": 600,
            "languages": ["en"],
            "tasks": ["image_text_matching", "zero_shot_classification", 
                     "text_image_retrieval", "multimodal_embedding"],
        },
        "openai/clip-vit-base-patch16": {
            "description": "CLIP base model with ViT-B/16 image encoder",
            "image_size": (224, 224),
            "embedding_dim": 512,
            "parameters": 151_000_000,
            "performance": "good",
            "memory_mb": 600,
            "languages": ["en"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
        "openai/clip-vit-large-patch14": {
            "description": "CLIP large model with ViT-L/14 image encoder",
            "image_size": (224, 224),
            "embedding_dim": 768,
            "parameters": 428_000_000,
            "performance": "excellent",
            "memory_mb": 1600,
            "languages": ["en"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
        "openai/clip-vit-large-patch14-336": {
            "description": "CLIP large model fine-tuned on 336px images",
            "image_size": (336, 336),
            "embedding_dim": 768,
            "parameters": 428_000_000,
            "performance": "excellent",
            "memory_mb": 1600,
            "languages": ["en"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
        "openai/clip-vit-huge-patch14": {
            "description": "CLIP huge model with ViT-H/14 image encoder",
            "image_size": (224, 224),
            "embedding_dim": 1024,
            "parameters": 958_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 3600,
            "languages": ["en"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
        # Multilingual CLIP models
        "openai/clip-vit-base-patch32-multilingual-v1": {
            "description": "Multilingual CLIP base model",
            "image_size": (224, 224),
            "embedding_dim": 512,
            "parameters": 151_000_000,
            "performance": "good",
            "memory_mb": 600,
            "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
        "openai/clip-vit-large-patch14-multilingual-v1": {
            "description": "Multilingual CLIP large model",
            "image_size": (224, 224),
            "embedding_dim": 768,
            "parameters": 428_000_000,
            "performance": "excellent",
            "memory_mb": 1600,
            "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
        # Community models
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
            "description": "LAION CLIP huge model trained on LAION-2B",
            "image_size": (224, 224),
            "embedding_dim": 1024,
            "parameters": 958_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 3600,
            "languages": ["en"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
        "laion/CLIP-ViT-L-14-laion2B-s32B-b79K": {
            "description": "LAION CLIP large model trained on LAION-2B",
            "image_size": (224, 224),
            "embedding_dim": 768,
            "parameters": 428_000_000,
            "performance": "excellent",
            "memory_mb": 1600,
            "languages": ["en"],
            "tasks": ["image_text_matching", "zero_shot_classification",
                     "text_image_retrieval", "multimodal_embedding"],
        },
    }
    
    # Default prompt templates for zero-shot classification
    PROMPT_TEMPLATES = {
        "default": "a photo of a {}.",
        "high_detail": "a detailed photo of a {}.",
        "art": "a painting of a {}.",
        "sketch": "a sketch of a {}.",
        "cartoon": "a cartoon of a {}.",
        "professional": "a professional photo of a {}.",
        "amateur": "an amateur photo of a {}.",
        "closeup": "a close-up photo of a {}.",
        "wide": "a wide-angle photo of a {}.",
        "indoor": "an indoor photo of a {}.",
        "outdoor": "an outdoor photo of a {}.",
        "day": "a daytime photo of a {}.",
        "night": "a nighttime photo of a {}.",
    }
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        task_type: MultimodalTaskType = MultimodalTaskType.IMAGE_TEXT_MATCHING,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize CLIP model.
        
        Args:
            model_name: Name of the CLIP model
            task_type: Type of multimodal task
            device: Device to run model on
            model_kwargs: Additional arguments for model initialization
            processor_kwargs: Additional arguments for processor initialization
            **kwargs: Additional arguments
        """
        super().__init__(model_name, task_type, device, **kwargs)
        
        # CLIP-specific initialization
        self.model_kwargs = model_kwargs or {}
        self.processor_kwargs = processor_kwargs or {}
        
        # Get model info
        model_info = self.MODEL_INFO.get(model_name, {})
        self.image_size = model_info.get("image_size", (224, 224))
        self.embedding_dim = model_info.get("embedding_dim", 512)
        
        # Check if multilingual
        self._is_multilingual = "multilingual" in model_name.lower()
        
        # Default processing parameters
        self._default_params = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        
        logger.info(f"Initialized CLIP model: {model_name} "
                   f"(Multilingual: {self._is_multilingual})")
    
    def load(self) -> None:
        """
        Load the CLIP model and processor.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading CLIP model: {self.model_name}")
            
            # Update device if auto
            if self._device == ModelDevice.AUTO:
                self._device = ModelDevice.get_best_device()
            
            # Load processor
            logger.info(f"Loading CLIP processor for {self.model_name}")
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                **self.processor_kwargs
            )
            
            # Load model with memory optimizations if needed
            self.model_kwargs.setdefault("torch_dtype", torch.float32)
            
            # Handle large models
            model_info = self.MODEL_INFO.get(self.model_name, {})
            memory_mb = model_info.get("memory_mb", 600)
            
            if self._device == ModelDevice.CUDA:
                try:
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_mb = free_memory / (1024 * 1024)
                    
                    if free_memory_mb < memory_mb:
                        logger.warning(
                            f"GPU memory may be insufficient: {free_memory_mb:.0f}MB available, "
                            f"{memory_mb:.0f}MB recommended. Using float16."
                        )
                        self.model_kwargs["torch_dtype"] = torch.float16
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {e}")
            
            logger.info(f"Loading CLIP model for {self.model_name}")
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                **self.model_kwargs
            )
            
            # Move to device
            self.model.to(self._device)
            self.model.eval()  # Set to evaluation mode
            
            self._is_loaded = True
            
            logger.info(f"Successfully loaded CLIP model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load CLIP model {self.model_name}: {e}")
    
    def process(
        self,
        inputs: MultimodalInput,
        task_type: Optional[MultimodalTaskType] = None,
        **kwargs
    ) -> MultimodalResult:
        """
        Process multimodal inputs with CLIP.
        
        Args:
            inputs: Multimodal input data
            task_type: Override task type for this request
            **kwargs: Additional processing arguments
            
        Returns:
            MultimodalResult containing the processed result
            
        Raises:
            ValueError: If model is not loaded or inputs are invalid
            RuntimeError: If processing fails
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        task_type = task_type or self.task_type
        
        try:
            start_time = time.time()
            
            # Determine processing method based on task type
            if task_type == MultimodalTaskType.IMAGE_TEXT_MATCHING:
                result = self._process_image_text_matching(inputs, **kwargs)
            elif task_type == MultimodalTaskType.TEXT_IMAGE_RETRIEVAL:
                result = self._process_text_image_retrieval(inputs, **kwargs)
            elif task_type == MultimodalTaskType.IMAGE_TEXT_RETRIEVAL:
                result = self._process_image_text_retrieval(inputs, **kwargs)
            elif task_type == MultimodalTaskType.MULTIMODAL_CLASSIFICATION:
                result = self._process_zero_shot_classification(inputs, **kwargs)
            elif task_type == MultimodalTaskType.MULTIMODAL_EMBEDDING:
                result = self._process_multimodal_embeddings(inputs, **kwargs)
            else:
                raise ValueError(f"Unsupported task type for CLIP: {task_type}")
            
            # Update performance stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._processing_times.append(processing_time_ms)
            self._total_requests_processed += 1
            
            # Add processing time to result
            result.processing_time_ms = processing_time_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process inputs with CLIP: {e}")
            raise RuntimeError(f"Failed to process inputs with CLIP: {e}")
    
    def _process_image_text_matching(
        self,
        inputs: MultimodalInput,
        **kwargs
    ) -> MultimodalResult:
        """
        Compute similarity scores between images and texts.
        
        Args:
            inputs: Must contain both images and texts
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with similarity scores
        """
        if not inputs.images or not inputs.texts:
            raise ValueError("Both images and texts are required for image-text matching")
        
        # Load images
        pil_images = []
        for img_input in inputs.images:
            pil_image = self._load_image(
                img_input.image,
                target_size=self.image_size,
                format=img_input.format
            )
            pil_images.append(pil_image)
        
        # Extract texts
        texts = []
        for text_input in inputs.texts:
            if text_input.format != "string":
                raise ValueError("Only string format is supported for text inputs")
            texts.append(text_input.text)
        
        # Process inputs
        inputs_processed = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            **kwargs
        )
        
        # Move to device
        if self._device.startswith("cuda"):
            inputs_processed = {k: v.to(self._device) for k, v in inputs_processed.items()}
        
        # Compute similarity
        with torch.no_grad():
            outputs = self.model(**inputs_processed)
            
            # Get image and text features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity matrix
            similarity = (image_features @ text_features.T) * 100.0  # Scale to 0-100
            similarity = similarity.cpu().numpy()
        
        # Prepare result
        result_data = {
            "similarity_matrix": similarity.tolist(),
            "image_features": image_features.cpu().numpy().tolist(),
            "text_features": text_features.cpu().numpy().tolist(),
        }
        
        # For single image-text pair, return simple score
        if len(pil_images) == 1 and len(texts) == 1:
            result_data["similarity_score"] = float(similarity[0][0])
        
        return MultimodalResult(
            result=result_data,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.IMAGE_TEXT_MATCHING,
            input_info={
                "num_images": len(pil_images),
                "num_texts": len(texts),
                "image_sizes": [img.size for img in pil_images],
            },
            embeddings={
                "image": image_features.cpu().numpy(),
                "text": text_features.cpu().numpy(),
            },
            metadata={
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
                "is_multilingual": self._is_multilingual,
            }
        )
    
    def _process_zero_shot_classification(
        self,
        inputs: MultimodalInput,
        class_names: Optional[List[str]] = None,
        prompt_template: str = "default",
        **kwargs
    ) -> MultimodalResult:
        """
        Perform zero-shot image classification.
        
        Args:
            inputs: Must contain images
            class_names: List of class names to classify against
            prompt_template: Template to use for prompts
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with classification results
        """
        if not inputs.images:
            raise ValueError("Images are required for zero-shot classification")
        
        # Use provided class names or extract from text inputs
        if class_names is None:
            if not inputs.texts:
                raise ValueError("Either class_names or text inputs must be provided")
            class_names = [text_input.text for text_input in inputs.texts]
        
        # Load images
        pil_images = []
        for img_input in inputs.images:
            pil_image = self._load_image(
                img_input.image,
                target_size=self.image_size,
                format=img_input.format
            )
            pil_images.append(pil_image)
        
        # Create prompts using template
        template = self.PROMPT_TEMPLATES.get(prompt_template, self.PROMPT_TEMPLATES["default"])
        prompts = [template.format(class_name) for class_name in class_names]
        
        # Process inputs
        inputs_processed = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            **kwargs
        )
        
        # Move to device
        if self._device.startswith("cuda"):
            inputs_processed = {k: v.to(self._device) for k, v in inputs_processed.items()}
        
        # Compute classification scores
        with torch.no_grad():
            outputs = self.model(**inputs_processed)
            
            # Get image and text features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (logits)
            logits_per_image = (image_features @ text_features.T) * 100.0
            logits_per_text = logits_per_image.T
            
            # Apply softmax to get probabilities
            probs_per_image = logits_per_image.softmax(dim=-1)
            probs_per_image = probs_per_image.cpu().numpy()
            
            # Get predictions
            predictions = probs_per_image.argmax(axis=-1)
            confidence_scores = probs_per_image.max(axis=-1)
        
        # Prepare classification results
        classification_results = []
        for i, (pred_idx, confidence) in enumerate(zip(predictions, confidence_scores)):
            class_name = class_names[pred_idx]
            all_probs = probs_per_image[i].tolist()
            
            classification_results.append({
                "image_index": i,
                "predicted_class": class_name,
                "confidence": float(confidence),
                "class_probabilities": dict(zip(class_names, all_probs)),
                "top_classes": [
                    {
                        "class": class_names[j],
                        "probability": float(all_probs[j])
                    }
                    for j in np.argsort(all_probs)[-5:][::-1]  # Top 5
                ]
            })
        
        return MultimodalResult(
            result=classification_results,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.MULTIMODAL_CLASSIFICATION,
            input_info={
                "num_images": len(pil_images),
                "num_classes": len(class_names),
                "prompt_template": prompt_template,
            },
            confidence=float(np.mean(confidence_scores)),
            metadata={
                "class_names": class_names,
                "prompts": prompts,
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
            }
        )
    
    def _process_text_image_retrieval(
        self,
        inputs: MultimodalInput,
        top_k: int = 5,
        **kwargs
    ) -> MultimodalResult:
        """
        Retrieve images based on text query.
        
        Args:
            inputs: Must contain both images and texts
            top_k: Number of top images to retrieve
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with retrieval results
        """
        if not inputs.images or not inputs.texts:
            raise ValueError("Both images and texts are required for text-image retrieval")
        
        # For text-image retrieval, we expect one query text
        if len(inputs.texts) != 1:
            logger.warning(f"Expected 1 text query, got {len(inputs.texts)}. Using first text.")
        
        query_text = inputs.texts[0].text
        
        # Load images
        pil_images = []
        image_metadata = []
        for i, img_input in enumerate(inputs.images):
            pil_image = self._load_image(
                img_input.image,
                target_size=self.image_size,
                format=img_input.format
            )
            pil_images.append(pil_image)
            image_metadata.append({
                "index": i,
                "metadata": img_input.metadata,
                "size": pil_image.size,
            })
        
        # Process query text with all images
        inputs_processed = self.processor(
            text=[query_text],
            images=pil_images,
            return_tensors="pt",
            padding=True,
            **kwargs
        )
        
        # Move to device
        if self._device.startswith("cuda"):
            inputs_processed = {k: v.to(self._device) for k, v in inputs_processed.items()}
        
        # Compute similarity
        with torch.no_grad():
            outputs = self.model(**inputs_processed)
            
            # Get image and text features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (text_features @ image_features.T) * 100.0
            similarity = similarity.cpu().numpy().flatten()
        
        # Get top-k matches
        top_indices = np.argsort(similarity)[-top_k:][::-1]
        
        # Prepare retrieval results
        retrieval_results = []
        for rank, idx in enumerate(top_indices, 1):
            retrieval_results.append({
                "rank": rank,
                "image_index": int(idx),
                "similarity_score": float(similarity[idx]),
                "metadata": image_metadata[idx]["metadata"],
                "image_size": image_metadata[idx]["size"],
            })
        
        return MultimodalResult(
            result=retrieval_results,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.TEXT_IMAGE_RETRIEVAL,
            input_info={
                "query_text": query_text,
                "num_images": len(pil_images),
                "top_k": top_k,
            },
            embeddings={
                "image": image_features.cpu().numpy(),
                "text": text_features.cpu().numpy(),
            },
            metadata={
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
            }
        )
    
    def _process_image_text_retrieval(
        self,
        inputs: MultimodalInput,
        top_k: int = 5,
        **kwargs
    ) -> MultimodalResult:
        """
        Retrieve texts based on image query.
        
        Args:
            inputs: Must contain both images and texts
            top_k: Number of top texts to retrieve
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with retrieval results
        """
        if not inputs.images or not inputs.texts:
            raise ValueError("Both images and texts are required for image-text retrieval")
        
        # For image-text retrieval, we expect one query image
        if len(inputs.images) != 1:
            logger.warning(f"Expected 1 query image, got {len(inputs.images)}. Using first image.")
        
        # Load query image
        query_image = self._load_image(
            inputs.images[0].image,
            target_size=self.image_size,
            format=inputs.images[0].format
        )
        
        # Extract texts
        texts = []
        text_metadata = []
        for i, text_input in enumerate(inputs.texts):
            if text_input.format != "string":
                raise ValueError("Only string format is supported for text inputs")
            texts.append(text_input.text)
            text_metadata.append({
                "index": i,
                "metadata": text_input.metadata,
                "text_preview": text_input.text[:100] + "..." if len(text_input.text) > 100 else text_input.text,
            })
        
        # Process query image with all texts
        inputs_processed = self.processor(
            text=texts,
            images=[query_image],
            return_tensors="pt",
            padding=True,
            **kwargs
        )
        
        # Move to device
        if self._device.startswith("cuda"):
            inputs_processed = {k: v.to(self._device) for k, v in inputs_processed.items()}
        
        # Compute similarity
        with torch.no_grad():
            outputs = self.model(**inputs_processed)
            
            # Get image and text features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (image_features @ text_features.T) * 100.0
            similarity = similarity.cpu().numpy().flatten()
        
        # Get top-k matches
        top_indices = np.argsort(similarity)[-top_k:][::-1]
        
        # Prepare retrieval results
        retrieval_results = []
        for rank, idx in enumerate(top_indices, 1):
            retrieval_results.append({
                "rank": rank,
                "text_index": int(idx),
                "similarity_score": float(similarity[idx]),
                "text": texts[idx],
                "metadata": text_metadata[idx]["metadata"],
                "text_preview": text_metadata[idx]["text_preview"],
            })
        
        return MultimodalResult(
            result=retrieval_results,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.IMAGE_TEXT_RETRIEVAL,
            input_info={
                "query_image_size": query_image.size,
                "num_texts": len(texts),
                "top_k": top_k,
            },
            embeddings={
                "image": image_features.cpu().numpy(),
                "text": text_features.cpu().numpy(),
            },
            metadata={
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
            }
        )
    
    def _process_multimodal_embeddings(
        self,
        inputs: MultimodalInput,
        **kwargs
    ) -> MultimodalResult:
        """
        Extract multimodal embeddings from inputs.
        
        Args:
            inputs: Can contain images, texts, or both
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with embeddings
        """
        if not inputs.images and not inputs.texts:
            raise ValueError("Either images or texts are required for embeddings")
        
        image_features = None
        text_features = None
        
        # Process images if present
        if inputs.images:
            pil_images = []
            for img_input in inputs.images:
                pil_image = self._load_image(
                    img_input.image,
                    target_size=self.image_size,
                    format=img_input.format
                )
                pil_images.append(pil_image)
            
            # Process images
            image_inputs = self.processor(
                images=pil_images,
                return_tensors="pt",
                **kwargs
            )
            
            # Move to device
            if self._device.startswith("cuda"):
                image_inputs = {k: v.to(self._device) for k, v in image_inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()
        
        # Process texts if present
        if inputs.texts:
            texts = []
            for text_input in inputs.texts:
                if text_input.format != "string":
                    raise ValueError("Only string format is supported for text inputs")
                texts.append(text_input.text)
            
            # Process texts
            text_inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                **kwargs
            )
            
            # Move to device
            if self._device.startswith("cuda"):
                text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy()
        
        # Prepare result
        result_data = {}
        if image_features is not None:
            result_data["image_embeddings"] = image_features.tolist()
        if text_features is not None:
            result_data["text_embeddings"] = text_features.tolist()
        
        return MultimodalResult(
            result=result_data,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.MULTIMODAL_EMBEDDING,
            input_info={
                "num_images": len(inputs.images) if inputs.images else 0,
                "num_texts": len(inputs.texts) if inputs.texts else 0,
            },
            embeddings={
                "image": image_features,
                "text": text_features,
            } if image_features is not None or text_features is not None else None,
            metadata={
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
                "embedding_dim": self.embedding_dim,
            }
        )
    
    def get_embeddings(
        self,
        images: Optional[List[ImageInput]] = None,
        texts: Optional[List[TextInput]] = None,
        normalize: bool = True,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for images and/or texts.
        
        Args:
            images: List of image inputs
            texts: List of text inputs
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with image and/or text embeddings
            
        Raises:
            ValueError: If neither images nor texts provided
        """
        if not images and not texts:
            raise ValueError("Either images or texts must be provided")
        
        # Create multimodal input
        multimodal_input = MultimodalInput(images=images, texts=texts)
        
        # Process to get embeddings
        result = self._process_multimodal_embeddings(multimodal_input, **kwargs)
        
        # Extract embeddings from result
        embeddings = {}
        if result.embeddings:
            if "image" in result.embeddings and result.embeddings["image"] is not None:
                embeddings["image"] = result.embeddings["image"]
            if "text" in result.embeddings and result.embeddings["text"] is not None:
                embeddings["text"] = result.embeddings["text"]
        
        return embeddings
    
    def compute_similarity(
        self,
        image: ImageInput,
        text: TextInput,
        **kwargs
    ) -> float:
        """
        Compute similarity score between an image and text.
        
        Args:
            image: Image input
            text: Text input
            **kwargs: Additional arguments
            
        Returns:
            Similarity score (0-100)
            
        Raises:
            ValueError: If model is not loaded
        """
        if not self._is_loaded:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load() first.")
        
        # Create multimodal input
        multimodal_input = MultimodalInput(images=[image], texts=[text])
        
        # Process similarity
        result = self._process_image_text_matching(multimodal_input, **kwargs)
        
        # Extract similarity score
        if isinstance(result.result, dict) and "similarity_score" in result.result:
            return result.result["similarity_score"]
        else:
            # Fallback: extract from matrix
            similarity_matrix = result.result.get("similarity_matrix", [[0]])
            return float(similarity_matrix[0][0])
    
    def classify_image(
        self,
        image: ImageInput,
        class_names: List[str],
        prompt_template: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Classify an image using zero-shot classification.
        
        Args:
            image: Image input
            class_names: List of possible classes
            prompt_template: Template to use for prompts
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with classification results
        """
        # Create text inputs from class names
        text_inputs = [TextInput(text=name) for name in class_names]
        
        # Create multimodal input
        multimodal_input = MultimodalInput(images=[image], texts=text_inputs)
        
        # Process classification
        result = self._process_zero_shot_classification(
            multimodal_input,
            class_names=class_names,
            prompt_template=prompt_template,
            **kwargs
        )
        
        # Return first result (single image)
        if isinstance(result.result, list) and len(result.result) > 0:
            return result.result[0]
        else:
            return {"error": "Classification failed"}
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info(f"Unloaded CLIP model {self.model_name}")
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the CLIP model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "vision_encoder": "ViT" if "vit" in self.model_name.lower() else "ResNet",
            "text_encoder": "Transformer",
            "tasks": model_info.get("tasks", []),
            "max_image_size": self.image_size,
            "embedding_dim": self.embedding_dim,
            "is_multilingual": self._is_multilingual,
            "supports_images": True,
            "supports_text": True,
            "supports_audio": False,
            "supports_video": False,
            "max_text_length": 77,  # CLIP tokenizer limit
            "multilingual": self._is_multilingual,
            "supported_languages": model_info.get("languages", ["en"]),
            "model_parameters": model_info.get("parameters", 0),
            "memory_requirements_mb": model_info.get("memory_mb", 600),
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
            # Extract from model name
            model_lower = self.model_name.lower()
            
            if "huge" in model_lower:
                return "clip-vit-huge"
            elif "large" in model_lower:
                if "336" in model_lower:
                    return "clip-vit-large-336"
                else:
                    return "clip-vit-large"
            elif "base" in model_lower:
                if "patch32" in model_lower:
                    return "clip-vit-base-32"
                elif "patch16" in model_lower:
                    return "clip-vit-base-16"
                else:
                    return "clip-vit-base"
            elif "multilingual" in model_lower:
                return "clip-multilingual"
            else:
                return "clip"
            
        except:
            return "unknown"
    
    def list_prompt_templates(self) -> List[str]:
        """
        List available prompt templates.
        
        Returns:
            List of template names
        """
        return list(self.PROMPT_TEMPLATES.keys())
    
    def add_prompt_template(self, name: str, template: str) -> None:
        """
        Add a custom prompt template.
        
        Args:
            name: Name of the template
            template: Template string with {} for class name
        """
        if "{}" not in template:
            logger.warning(f"Template '{name}' does not contain {{}} placeholder")
        
        self.PROMPT_TEMPLATES[name] = template
        logger.info(f"Added prompt template: {name}")