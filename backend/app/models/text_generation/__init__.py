"""
ImageBind model implementation.

ImageBind is a groundbreaking multimodal model that learns a joint embedding
space across six modalities: images, text, audio, depth, thermal, and IMU data.
It enables cross-modal retrieval and reasoning without explicit supervision.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .base import (
    BaseMultimodalModel, MultimodalInput, MultimodalResult,
    MultimodalTaskType, ImageInput, TextInput, AudioInput,
    ModelDevice, AudioFormat
)

logger = logging.getLogger(__name__)

try:
    from imagebind import data # type: ignore
    from imagebind.models import imagebind_model # type: ignore
    from imagebind.models.imagebind_model import ModalityType # type: ignore
    HAS_IMAGEBIND = True
except ImportError:
    HAS_IMAGEBIND = False
    logger.warning("ImageBind not installed. Install with: pip install imagebind")


class ImageBindModel(BaseMultimodalModel):
    """
    ImageBind model wrapper for WorldBrief360.
    
    ImageBind learns a joint embedding space across six modalities:
    1. Images (RGB)
    2. Text
    3. Audio
    4. Depth (3D point clouds)
    5. Thermal (infrared images)
    6. IMU (inertial measurement units)
    
    Capabilities:
    - Cross-modal retrieval across all six modalities
    - Zero-shot classification using any modality as query
    - Multimodal embeddings in a shared space
    - Audio-based image retrieval and vice versa
    """
    
    # ImageBind model variants and their specifications
    MODEL_INFO = {
        "imagebind_huge": {
            "description": "ImageBind huge model (recommended)",
            "embedding_dim": 1024,
            "parameters": 1_200_000_000,
            "performance": "state_of_the_art",
            "memory_mb": 4600,
            "modalities": ["image", "text", "audio", "depth", "thermal", "imu"],
            "tasks": ["cross_modal_retrieval", "zero_shot_classification", 
                     "multimodal_embedding", "modality_translation"],
        },
    }
    
    # Modality type mapping
    MODALITY_MAPPING = {
        "image": ModalityType.VISION,
        "text": ModalityType.TEXT,
        "audio": ModalityType.AUDIO,
        "depth": ModalityType.DEPTH,
        "thermal": ModalityType.THERMAL,
        "imu": ModalityType.IMU,
    }
    
    # Supported audio formats for ImageBind
    SUPPORTED_AUDIO_FORMATS = [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FILE]
    
    def __init__(
        self,
        model_name: str = "imagebind_huge",
        task_type: MultimodalTaskType = MultimodalTaskType.MULTIMODAL_EMBEDDING,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize ImageBind model.
        
        Args:
            model_name: Name of the ImageBind model
            task_type: Type of multimodal task
            device: Device to run model on
            model_kwargs: Additional arguments for model initialization
            **kwargs: Additional arguments
        """
        if not HAS_IMAGEBIND:
            raise ImportError(
                "ImageBind is not installed. "
                "Install with: pip install imagebind"
            )
        
        super().__init__(model_name, task_type, device, **kwargs)
        
        # ImageBind-specific initialization
        self.model_kwargs = model_kwargs or {}
        
        # Get model info
        model_info = self.MODEL_INFO.get(model_name, {})
        self.embedding_dim = model_info.get("embedding_dim", 1024)
        self.supported_modalities = model_info.get("modalities", ["image", "text", "audio"])
        
        # ImageBind specific settings
        self._modality_keys = {}
        self._data_module = data
        
        logger.info(f"Initialized ImageBind model: {model_name}")
    
    def load(self) -> None:
        """
        Load the ImageBind model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if not HAS_IMAGEBIND:
            raise ImportError(
                "ImageBind is not installed. "
                "Install with: pip install imagebind"
            )
        
        if self._is_loaded:
            logger.warning(f"Model {self.model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading ImageBind model: {self.model_name}")
            
            # Update device if auto
            if self._device == ModelDevice.AUTO:
                self._device = ModelDevice.get_best_device()
            
            # Load model
            logger.info(f"Loading ImageBind model for {self.model_name}")
            
            # ImageBind has specific device handling
            device = "cuda" if self._device == ModelDevice.CUDA else self._device
            
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            self.model.eval()
            self.model.to(device)
            
            self._is_loaded = True
            
            logger.info(f"Successfully loaded ImageBind model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load ImageBind model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load ImageBind model {self.model_name}: {e}")
    
    def process(
        self,
        inputs: MultimodalInput,
        task_type: Optional[MultimodalTaskType] = None,
        **kwargs
    ) -> MultimodalResult:
        """
        Process multimodal inputs with ImageBind.
        
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
            if task_type == MultimodalTaskType.MULTIMODAL_EMBEDDING:
                result = self._process_multimodal_embeddings(inputs, **kwargs)
            elif task_type == MultimodalTaskType.TEXT_IMAGE_RETRIEVAL:
                result = self._process_cross_modal_retrieval(inputs, query_modality="text", **kwargs)
            elif task_type == MultimodalTaskType.IMAGE_TEXT_RETRIEVAL:
                result = self._process_cross_modal_retrieval(inputs, query_modality="image", **kwargs)
            elif task_type == MultimodalTaskType.MULTIMODAL_CLASSIFICATION:
                result = self._process_zero_shot_classification(inputs, **kwargs)
            else:
                # Default to cross-modal retrieval
                result = self._process_cross_modal_retrieval(inputs, **kwargs)
            
            # Update performance stats
            processing_time_ms = (time.time() - start_time) * 1000
            self._processing_times.append(processing_time_ms)
            self._total_requests_processed += 1
            
            # Add processing time to result
            result.processing_time_ms = processing_time_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process inputs with ImageBind: {e}")
            raise RuntimeError(f"Failed to process inputs with ImageBind: {e}")
    
    def _process_multimodal_embeddings(
        self,
        inputs: MultimodalInput,
        normalize: bool = True,
        **kwargs
    ) -> MultimodalResult:
        """
        Extract multimodal embeddings from inputs.
        
        Args:
            inputs: Can contain images, texts, audios
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with embeddings
        """
        if not inputs.images and not inputs.texts and not inputs.audios:
            raise ValueError("At least one modality (images, texts, or audios) is required")
        
        inputs_dict = {}
        
        # Process images if present
        if inputs.images:
            image_paths = []
            for img_input in inputs.images:
                # Save image to temporary file for ImageBind processing
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    pil_image = self._load_image(
                        img_input.image,
                        target_size=None,  # ImageBind handles resizing
                        format=img_input.format
                    )
                    pil_image.save(tmp.name)
                    image_paths.append(tmp.name)
            
            inputs_dict["vision"] = image_paths
        
        # Process texts if present
        if inputs.texts:
            texts = []
            for text_input in inputs.texts:
                if text_input.format != "string":
                    raise ValueError("Only string format is supported for text inputs")
                texts.append(text_input.text)
            
            inputs_dict["text"] = texts
        
        # Process audios if present
        if inputs.audios:
            audio_paths = []
            for audio_input in inputs.audios:
                # ImageBind expects audio files
                if audio_input.format in [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FILE]:
                    if isinstance(audio_input.audio, (str, Path)):
                        audio_paths.append(str(audio_input.audio))
                    else:
                        # Save audio to temporary file
                        import tempfile
                        import soundfile as sf
                        
                        audio_array, sample_rate = self._load_audio(
                            audio_input.audio,
                            target_sample_rate=16000,  # ImageBind expects 16kHz
                            format=audio_input.format
                        )
                        
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            sf.write(tmp.name, audio_array, sample_rate)
                            audio_paths.append(tmp.name)
                else:
                    raise ValueError(f"Audio format {audio_input.format} not supported for ImageBind")
            
            inputs_dict["audio"] = audio_paths
        
        # Load and transform data
        inputs_transformed = self._load_and_transform_data(inputs_dict)
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs_transformed = {k: v.to(device) if hasattr(v, 'to') else v 
                             for k, v in inputs_transformed.items()}
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(inputs_transformed)
        
        # Prepare result
        result_data = {}
        embeddings_dict = {}
        
        for modality, emb in embeddings.items():
            if emb is not None:
                # Normalize if requested
                if normalize:
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                emb_np = emb.cpu().numpy()
                result_data[f"{modality}_embeddings"] = emb_np.tolist()
                embeddings_dict[modality] = emb_np
        
        # Count processed items
        input_info = {
            "num_images": len(inputs.images) if inputs.images else 0,
            "num_texts": len(inputs.texts) if inputs.texts else 0,
            "num_audios": len(inputs.audios) if inputs.audios else 0,
        }
        
        return MultimodalResult(
            result=result_data,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.MULTIMODAL_EMBEDDING,
            input_info=input_info,
            embeddings=embeddings_dict,
            metadata={
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
                "normalized": normalize,
                "embedding_dim": self.embedding_dim,
            }
        )
    
    def _process_cross_modal_retrieval(
        self,
        inputs: MultimodalInput,
        query_modality: str = "text",
        target_modality: str = "image",
        top_k: int = 5,
        **kwargs
    ) -> MultimodalResult:
        """
        Perform cross-modal retrieval.
        
        Args:
            inputs: Must contain both query and target modalities
            query_modality: Modality to query with (image, text, audio)
            target_modality: Modality to retrieve (image, text, audio)
            top_k: Number of top results to retrieve
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with retrieval results
            
        Raises:
            ValueError: If required modalities not provided
        """
        # Validate modalities
        if query_modality not in ["image", "text", "audio"]:
            raise ValueError(f"Unsupported query modality: {query_modality}")
        if target_modality not in ["image", "text", "audio"]:
            raise ValueError(f"Unsupported target modality: {target_modality}")
        
        # Get query inputs
        if query_modality == "image":
            if not inputs.images:
                raise ValueError(f"Images required for {query_modality} query")
            query_items = inputs.images
        elif query_modality == "text":
            if not inputs.texts:
                raise ValueError(f"Texts required for {query_modality} query")
            query_items = inputs.texts
        elif query_modality == "audio":
            if not inputs.audios:
                raise ValueError(f"Audios required for {query_modality} query")
            query_items = inputs.audios
        
        # Get target inputs
        if target_modality == "image":
            if not inputs.images:
                raise ValueError(f"Images required for {target_modality} target")
            target_items = inputs.images
        elif target_modality == "text":
            if not inputs.texts:
                raise ValueError(f"Texts required for {target_modality} target")
            target_items = inputs.texts
        elif target_modality == "audio":
            if not inputs.audios:
                raise ValueError(f"Audios required for {target_modality} target")
            target_items = inputs.audios
        
        # For simplicity, use first query item
        query_item = query_items[0]
        
        # Get embeddings for query and all targets
        query_embeddings = self._get_single_embedding(query_item, query_modality)
        target_embeddings = []
        target_metadata = []
        
        for idx, target_item in enumerate(target_items):
            emb = self._get_single_embedding(target_item, target_modality)
            target_embeddings.append(emb)
            
            # Store metadata
            if target_modality == "image":
                target_metadata.append({
                    "index": idx,
                    "metadata": target_item.metadata,
                    "type": "image",
                })
            elif target_modality == "text":
                target_metadata.append({
                    "index": idx,
                    "metadata": target_item.metadata,
                    "type": "text",
                    "text_preview": target_item.text[:100] + "..." 
                    if len(target_item.text) > 100 else target_item.text,
                })
            elif target_modality == "audio":
                target_metadata.append({
                    "index": idx,
                    "metadata": target_item.metadata,
                    "type": "audio",
                })
        
        # Convert to numpy arrays
        query_emb_np = query_embeddings.cpu().numpy() if hasattr(query_embeddings, 'cpu') else query_embeddings
        target_emb_np = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb for emb in target_embeddings])
        
        # Normalize embeddings
        query_emb_np = query_emb_np / np.linalg.norm(query_emb_np)
        target_emb_np = target_emb_np / np.linalg.norm(target_emb_np, axis=1, keepdims=True)
        
        # Compute similarity scores
        similarity = (query_emb_np @ target_emb_np.T).flatten()
        
        # Get top-k matches
        top_indices = np.argsort(similarity)[-top_k:][::-1]
        
        # Prepare retrieval results
        retrieval_results = []
        for rank, idx in enumerate(top_indices, 1):
            result_item = {
                "rank": rank,
                f"{target_modality}_index": int(idx),
                "similarity_score": float(similarity[idx]),
                "metadata": target_metadata[idx]["metadata"],
                "type": target_metadata[idx]["type"],
            }
            
            # Add modality-specific info
            if target_modality == "text":
                result_item["text_preview"] = target_metadata[idx]["text_preview"]
            
            retrieval_results.append(result_item)
        
        # Prepare input info
        input_info = {
            "query_modality": query_modality,
            "target_modality": target_modality,
            f"num_query_{query_modality}s": 1,
            f"num_target_{target_modality}s": len(target_items),
            "top_k": top_k,
        }
        
        # Add query info
        if query_modality == "image":
            pil_image = self._load_image(query_item.image, format=query_item.format)
            input_info["query_image_size"] = pil_image.size
        elif query_modality == "text":
            input_info["query_text"] = query_item.text[:200] + "..." if len(query_item.text) > 200 else query_item.text
        
        return MultimodalResult(
            result=retrieval_results,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.TEXT_IMAGE_RETRIEVAL 
            if query_modality == "text" and target_modality == "image"
            else MultimodalTaskType.IMAGE_TEXT_RETRIEVAL 
            if query_modality == "image" and target_modality == "text"
            else MultimodalTaskType.MULTIMODAL_EMBEDDING,
            input_info=input_info,
            embeddings={
                "query": query_emb_np,
                "target": target_emb_np,
            },
            metadata={
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
                "cross_modal_retrieval": True,
            }
        )
    
    def _process_zero_shot_classification(
        self,
        inputs: MultimodalInput,
        class_names: Optional[List[str]] = None,
        query_modality: str = "image",
        **kwargs
    ) -> MultimodalResult:
        """
        Perform zero-shot classification using any modality as query.
        
        Args:
            inputs: Must contain query items and optionally text class names
            class_names: List of class names to classify against
            query_modality: Modality of query items (image, text, audio)
            **kwargs: Additional arguments
            
        Returns:
            MultimodalResult with classification results
        """
        if query_modality not in ["image", "text", "audio"]:
            raise ValueError(f"Unsupported query modality: {query_modality}")
        
        # Get query items
        if query_modality == "image":
            if not inputs.images:
                raise ValueError(f"Images required for {query_modality} query")
            query_items = inputs.images
        elif query_modality == "text":
            if not inputs.texts:
                raise ValueError(f"Texts required for {query_modality} query")
            query_items = inputs.texts
        elif query_modality == "audio":
            if not inputs.audios:
                raise ValueError(f"Audios required for {query_modality} query")
            query_items = inputs.audios
        
        # Use provided class names or extract from text inputs
        if class_names is None:
            if not inputs.texts:
                raise ValueError("Either class_names or text inputs must be provided")
            class_names = [text_input.text for text_input in inputs.texts]
        
        # Create text inputs for class names
        class_texts = [f"a photo of {name}" for name in class_names]
        
        # Get embeddings for all queries and class texts
        query_embeddings = []
        for query_item in query_items:
            emb = self._get_single_embedding(query_item, query_modality)
            query_embeddings.append(emb)
        
        class_embeddings = []
        for class_text in class_texts:
            text_input = TextInput(text=class_text)
            emb = self._get_single_embedding(text_input, "text")
            class_embeddings.append(emb)
        
        # Convert to numpy
        query_emb_np = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb 
                                for emb in query_embeddings])
        class_emb_np = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb 
                                for emb in class_embeddings])
        
        # Normalize embeddings
        query_emb_np = query_emb_np / np.linalg.norm(query_emb_np, axis=1, keepdims=True)
        class_emb_np = class_emb_np / np.linalg.norm(class_emb_np, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity = query_emb_np @ class_emb_np.T
        
        # Get predictions and confidence scores
        predictions = similarity.argmax(axis=1)
        confidence_scores = similarity.max(axis=1)
        
        # Prepare classification results
        classification_results = []
        for i, (pred_idx, confidence) in enumerate(zip(predictions, confidence_scores)):
            class_name = class_names[pred_idx]
            all_probs = similarity[i].tolist()
            
            # Apply softmax to get probabilities
            import scipy.special
            probs = scipy.special.softmax(similarity[i])
            
            classification_results.append({
                "query_index": i,
                "query_modality": query_modality,
                "predicted_class": class_name,
                "confidence": float(confidence),
                "class_probabilities": dict(zip(class_names, probs.tolist())),
                "top_classes": [
                    {
                        "class": class_names[j],
                        "probability": float(probs[j]),
                        "similarity": float(similarity[i][j])
                    }
                    for j in np.argsort(probs)[-5:][::-1]  # Top 5
                ]
            })
        
        return MultimodalResult(
            result=classification_results,
            model_name=self.model_name,
            model_version=self.get_version(),
            task_type=MultimodalTaskType.MULTIMODAL_CLASSIFICATION,
            input_info={
                f"num_query_{query_modality}s": len(query_items),
                "num_classes": len(class_names),
                "query_modality": query_modality,
            },
            confidence=float(np.mean(confidence_scores)),
            embeddings={
                "query": query_emb_np,
                "classes": class_emb_np,
            },
            metadata={
                "class_names": class_names,
                "class_prompts": class_texts,
                "model_info": self.MODEL_INFO.get(self.model_name, {}),
            }
        )
    
    def _get_single_embedding(
        self,
        item: Union[ImageInput, TextInput, AudioInput],
        modality: str
    ) -> torch.Tensor:
        """
        Get embedding for a single item.
        
        Args:
            item: Input item
            modality: Type of modality
            
        Returns:
            Embedding tensor
        """
        # Create inputs dict for single item
        inputs_dict = {}
        
        if modality == "image":
            # Save image to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_image = self._load_image(
                    item.image,
                    target_size=None,
                    format=item.format
                )
                pil_image.save(tmp.name)
                inputs_dict["vision"] = [tmp.name]
        
        elif modality == "text":
            if item.format != "string":
                raise ValueError("Only string format is supported for text inputs")
            inputs_dict["text"] = [item.text]
        
        elif modality == "audio":
            # ImageBind expects audio files
            if item.format in [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FILE]:
                if isinstance(item.audio, (str, Path)):
                    inputs_dict["audio"] = [str(item.audio)]
                else:
                    # Save audio to temporary file
                    import tempfile
                    import soundfile as sf
                    
                    audio_array, sample_rate = self._load_audio(
                        item.audio,
                        target_sample_rate=16000,
                        format=item.format
                    )
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        sf.write(tmp.name, audio_array, sample_rate)
                        inputs_dict["audio"] = [tmp.name]
            else:
                raise ValueError(f"Audio format {item.format} not supported for ImageBind")
        
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Load and transform data
        inputs_transformed = self._load_and_transform_data(inputs_dict)
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs_transformed = {k: v.to(device) if hasattr(v, 'to') else v 
                             for k, v in inputs_transformed.items()}
        
        # Get embedding
        with torch.no_grad():
            embeddings = self.model(inputs_transformed)
        
        # Extract the specific modality embedding
        modality_key = self._get_modality_key(modality)
        embedding = embeddings.get(modality_key)
        
        if embedding is None:
            raise ValueError(f"Failed to get embedding for modality: {modality}")
        
        return embedding[0]  # Return first (and only) item
    
    def _load_and_transform_data(self, inputs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Load and transform data for ImageBind processing.
        
        Args:
            inputs_dict: Dictionary with modality keys and input data
            
        Returns:
            Dictionary with transformed tensors
        """
        transformed = {}
        
        if "vision" in inputs_dict:
            image_paths = inputs_dict["vision"]
            transformed[ModalityType.VISION] = self._data_module.load_and_transform_vision_data(
                image_paths, device="cpu"
            )
        
        if "text" in inputs_dict:
            texts = inputs_dict["text"]
            transformed[ModalityType.TEXT] = self._data_module.load_and_transform_text(
                texts, device="cpu"
            )
        
        if "audio" in inputs_dict:
            audio_paths = inputs_dict["audio"]
            transformed[ModalityType.AUDIO] = self._data_module.load_and_transform_audio_data(
                audio_paths, device="cpu"
            )
        
        return transformed
    
    def _get_modality_key(self, modality: str) -> ModalityType:
        """
        Get ImageBind modality key for a modality string.
        
        Args:
            modality: Modality string (image, text, audio, etc.)
            
        Returns:
            ImageBind ModalityType
        """
        if modality not in self.MODALITY_MAPPING:
            raise ValueError(f"Unknown modality: {modality}")
        
        return self.MODALITY_MAPPING[modality]
    
    def get_embeddings(
        self,
        images: Optional[List[ImageInput]] = None,
        texts: Optional[List[TextInput]] = None,
        audios: Optional[List[AudioInput]] = None,
        normalize: bool = True,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple modalities.
        
        Args:
            images: List of image inputs
            texts: List of text inputs
            audios: List of audio inputs
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with embeddings for each modality
        """
        if not images and not texts and not audios:
            raise ValueError("At least one modality (images, texts, or audios) must be provided")
        
        # Create multimodal input
        multimodal_input = MultimodalInput(
            images=images,
            texts=texts,
            audios=audios
        )
        
        # Process to get embeddings
        result = self._process_multimodal_embeddings(
            multimodal_input,
            normalize=normalize,
            **kwargs
        )
        
        # Extract embeddings from result
        embeddings = {}
        if result.embeddings:
            for modality, emb in result.embeddings.items():
                embeddings[modality] = emb
        
        return embeddings
    
    def compute_cross_modal_similarity(
        self,
        source_item: Union[ImageInput, TextInput, AudioInput],
        target_item: Union[ImageInput, TextInput, AudioInput],
        source_modality: str,
        target_modality: str,
        **kwargs
    ) -> float:
        """
        Compute similarity between items from different modalities.
        
        Args:
            source_item: Source item
            target_item: Target item
            source_modality: Modality of source item
            target_modality: Modality of target item
            **kwargs: Additional arguments
            
        Returns:
            Similarity score (normalized cosine similarity)
        """
        # Get embeddings
        source_emb = self._get_single_embedding(source_item, source_modality)
        target_emb = self._get_single_embedding(target_item, target_modality)
        
        # Convert to numpy
        source_np = source_emb.cpu().numpy() if hasattr(source_emb, 'cpu') else source_emb
        target_np = target_emb.cpu().numpy() if hasattr(target_emb, 'cpu') else target_emb
        
        # Normalize and compute similarity
        source_np = source_np / np.linalg.norm(source_np)
        target_np = target_np / np.linalg.norm(target_np)
        
        similarity = float(np.dot(source_np, target_np))
        
        return similarity
    
    def retrieve_by_modality(
        self,
        query_item: Union[ImageInput, TextInput, AudioInput],
        target_items: List[Union[ImageInput, TextInput, AudioInput]],
        query_modality: str,
        target_modality: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve target items based on a query from a different modality.
        
        Args:
            query_item: Query item
            target_items: List of target items
            query_modality: Modality of query item
            target_modality: Modality of target items
            top_k: Number of top results to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieval results
        """
        # Create inputs for retrieval
        if query_modality == "image":
            query_images = [query_item]
            query_texts = None
            query_audios = None
        elif query_modality == "text":
            query_images = None
            query_texts = [query_item]
            query_audios = None
        elif query_modality == "audio":
            query_images = None
            query_texts = None
            query_audios = [query_item]
        
        if target_modality == "image":
            target_images = target_items
            target_texts = None
            target_audios = None
        elif target_modality == "text":
            target_images = None
            target_texts = target_items
            target_audios = None
        elif target_modality == "audio":
            target_images = None
            target_texts = None
            target_audios = target_items
        
        # Create multimodal input
        multimodal_input = MultimodalInput(
            images=query_images or target_images,
            texts=query_texts or target_texts,
            audios=query_audios or target_audios
        )
        
        # Process retrieval
        result = self._process_cross_modal_retrieval(
            multimodal_input,
            query_modality=query_modality,
            target_modality=target_modality,
            top_k=top_k,
            **kwargs
        )
        
        return result.result if isinstance(result.result, list) else []
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info(f"Unloaded ImageBind model {self.model_name}")
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the ImageBind model.
        
        Returns:
            Dictionary with model capabilities
        """
        model_info = self.MODEL_INFO.get(self.model_name, {}).copy()
        
        capabilities = {
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "modalities": model_info.get("modalities", []),
            "tasks": model_info.get("tasks", []),
            "supports_images": "image" in model_info.get("modalities", []),
            "supports_text": "text" in model_info.get("modalities", []),
            "supports_audio": "audio" in model_info.get("modalities", []),
            "supports_depth": "depth" in model_info.get("modalities", []),
            "supports_thermal": "thermal" in model_info.get("modalities", []),
            "supports_imu": "imu" in model_info.get("modalities", []),
            "supports_video": False,
            "max_image_size": None,  # ImageBind handles various sizes
            "max_text_length": 77,  # Similar to CLIP
            "max_audio_length": None,
            "multilingual": False,  # ImageBind is English-focused
            "model_parameters": model_info.get("parameters", 0),
            "memory_requirements_mb": model_info.get("memory_mb", 4600),
            "cross_modal_alignment": True,
            "zero_shot_capable": True,
        }
        
        return capabilities
    
    def get_version(self) -> str:
        """
        Get model version.
        
        Returns:
            Model version string
        """
        return "imagebind-huge"
    
    def get_supported_modalities(self) -> List[str]:
        """
        Get list of supported modalities.
        
        Returns:
            List of modality names
        """
        model_info = self.MODEL_INFO.get(self.model_name, {})
        return model_info.get("modalities", ["image", "text", "audio"])
    
    def is_modality_supported(self, modality: str) -> bool:
        """
        Check if a modality is supported.
        
        Args:
            modality: Modality name
            
        Returns:
            True if supported, False otherwise
        """
        supported = self.get_supported_modalities()
        return modality.lower() in [m.lower() for m in supported]