"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  TrafficSignDetector,
  DetectedSign,
  ProcessingSteps,
} from "@/lib/detect";

export default function Home() {
  const [detector] = useState(() => new TrafficSignDetector());
  const [isLoading, setIsLoading] = useState(false);
  const [isDetectorReady, setIsDetectorReady] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [detections, setDetections] = useState<DetectedSign[]>([]);
  const [originalDetections, setOriginalDetections] = useState<DetectedSign[]>([]);
  const [processingSteps, setProcessingSteps] = useState<
    ProcessingSteps[] | null
  >(null);
  const [scaleFactor, setScaleFactor] = useState<number>(1);
  const [detectedImageUrl, setDetectedImageUrl] = useState<string | null>(null);
  const [croppedImageUrl, setCroppedImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Model selection state
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  // Confidence filtering state
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5); // 50% default

  // UI state
  const [isProcessingStepsCollapsed, setIsProcessingStepsCollapsed] = useState<boolean>(true); // Collapsed by default

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // Load available models from backend
  const loadAvailableModels = useCallback(async () => {
    try {
      setIsLoadingModels(true);
      const models = await detector.getAvailableModels();
      setAvailableModels(models);

      // Pre-select the first model if available
      if (models.length > 0) {
        setSelectedModels([models[0]]);
      }
    } catch (err) {
      console.warn("Failed to load models:", err);
    } finally {
      setIsLoadingModels(false);
    }
  }, [detector]);

  // Initialize OpenCV when component mounts
  const initializeDetector = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      await detector.initialize();
      setIsDetectorReady(true);

      // Load available models after detector is ready
      await loadAvailableModels();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to initialize detector"
      );
    } finally {
      setIsLoading(false);
    }
  }, [detector, loadAvailableModels]);

  // Handle model selection change
  const handleModelSelection = useCallback(
    (modelName: string, isSelected: boolean) => {
      setSelectedModels((prev) => {
        if (isSelected) {
          return [...prev, modelName];
        } else {
          return prev.filter((m) => m !== modelName);
        }
      });
    },
    []
  );

  // Select all models
  const selectAllModels = useCallback(() => {
    setSelectedModels([...availableModels]);
  }, [availableModels]);

  // Clear all selections
  const clearAllModels = useCallback(() => {
    setSelectedModels([]);
  }, []);

  const handleFileUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      if (!file.type.startsWith("image/")) {
        setError("Please select a valid image file");
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        setDetections([]);
        setProcessingSteps(null);
        setScaleFactor(1);
        setDetectedImageUrl(null);
        setCroppedImageUrl(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    },
    []
  );

  const detectTrafficSigns = useCallback(async () => {
    if (!imageRef.current || !isDetectorReady) return;

    if (selectedModels.length === 0) {
      setError("Please select at least one model for classification");
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Step 1: OpenCV detection
      console.log("🔍 Starting OpenCV detection...");
      const result = await detector.detectTrafficSignsWithSteps(
        imageRef.current
      );

      console.log(`🎯 Found ${result.detections.length} potential regions`);
      setProcessingSteps(result.processingSteps);
      setScaleFactor(result.scaleFactor);

      let finalDetections: DetectedSign[] = [];

      if (result.detections.length > 0) {
        // Step 2: AI Classification of detected regions with multiple models
        console.log(
          `🤖 Starting AI classification of detected regions with ${selectedModels.length} models...`
        );
        finalDetections = await detector.classifyWithMultipleModels(
          imageRef.current,
          result.detections,
          selectedModels
        );

        console.log(
          "✅ Multi-model region-based detection and classification complete!"
        );
      } else {
        // Step 2: Fallback - Classify entire image with multiple models
        console.log(
          `⚠️ No regions detected, trying full image classification with ${selectedModels.length} models...`
        );
        finalDetections = await detector.classifyEntireImageWithMultipleModels(
          imageRef.current,
          selectedModels
        );

        if (finalDetections.length > 0) {
          console.log("✅ Multi-model full image classification successful!");
        } else {
          console.log(
            "❌ No signs detected even with multi-model full image classification"
          );
        }
      }

      if (finalDetections.length > 0) {
        // Store original detections for real-time filtering
        setOriginalDetections(finalDetections);
        
        // Initial filtering will be handled by useEffect
        // This ensures consistency with real-time filtering
      } else {
        setDetections([]);
        setOriginalDetections([]);
        setError(
          `No traffic signs detected in the image (tried both region detection and full image classification with ${selectedModels.length} models)`
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
      console.error("❌ Detection error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [detector, isDetectorReady, selectedModels]);

  const downloadCroppedImage = useCallback(() => {
    if (!croppedImageUrl) return;

    const link = document.createElement("a");
    link.href = croppedImageUrl;
    link.download = "cropped_traffic_sign.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [croppedImageUrl]);

  const resetAll = useCallback(() => {
    setUploadedImage(null);
    setDetections([]);
    setOriginalDetections([]);
    setProcessingSteps(null);
    setScaleFactor(1);
    setDetectedImageUrl(null);
    setCroppedImageUrl(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  // Function to get comprehensive image information
  const getImageInfo = useCallback(() => {
    if (!imageRef.current) {
      alert('No image loaded');
      return;
    }

    const img = imageRef.current;
    const info = {
      // Original/Natural dimensions (from the actual image file)
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
      
      // Displayed dimensions (how it appears on screen)
      displayedWidth: img.width,
      displayedHeight: img.height,
      
      // CSS dimensions
      offsetWidth: img.offsetWidth,
      offsetHeight: img.offsetHeight,
      
      // Other properties
      src: img.src.substring(0, 50) + '...',
      complete: img.complete,
      crossOrigin: img.crossOrigin,
    };

    console.log('📏 Image Information:', info);
    
    const message = `
Image Dimensions:
• Original size: ${info.naturalWidth} × ${info.naturalHeight}px
• Displayed size: ${info.displayedWidth} × ${info.displayedHeight}px
• Offset size: ${info.offsetWidth} × ${info.offsetHeight}px
• Image loaded: ${info.complete}

Check console for full details.`;
    
    alert(message);
  }, []);

  // Real-time confidence filtering
  useEffect(() => {
    if (originalDetections.length > 0) {
      const filteredDetections = originalDetections.filter(detection => {
        const detectionConfidence = detection.confidence;
        const aiConfidence = detection.classification?.ai_confidence || 0;
        const maxConfidence = Math.max(detectionConfidence, aiConfidence);
        return maxConfidence >= confidenceThreshold;
      });

      setDetections(filteredDetections);

      // Update visualizations if we have an image
      if (imageRef.current && filteredDetections.length > 0) {
        // Redraw detections with filtered results
        const detectedCanvas = detector.drawDetections(
          imageRef.current,
          filteredDetections
        );
        setDetectedImageUrl(detectedCanvas.toDataURL());

        // Update cropped image with best filtered detection
        const croppedCanvas = detector.cropDetectedRegion(
          imageRef.current,
          filteredDetections[0]
        );
        setCroppedImageUrl(croppedCanvas.toDataURL());
      } else if (originalDetections.length > 0 && filteredDetections.length === 0) {
        // Clear visualizations if all detections are filtered out
        setDetectedImageUrl(null);
        setCroppedImageUrl(null);
      }

      console.log(`🔍 Real-time filter: ${originalDetections.length} → ${filteredDetections.length} detections (${(confidenceThreshold * 100).toFixed(0)}% threshold)`);
    }
  }, [confidenceThreshold, originalDetections, detector]);

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🚦 AI Traffic Sign Detection & Classification
          </h1>
          <p className="text-lg text-gray-600">
            Upload an image to detect traffic signs with OpenCV.js and classify
            them using PyTorch AI
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Combines computer vision region detection with deep learning
            classification (99.45% accuracy)
          </p>
        </div>

        {/* Initialize Detector */}
        {!isDetectorReady && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-4">
                Initialize Detection Engine
              </h2>
              <button
                onClick={initializeDetector}
                disabled={isLoading}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-6 py-3 rounded-lg font-medium transition-colors"
              >
                {isLoading ? "Loading OpenCV.js..." : "Initialize Detector"}
              </button>
            </div>
          </div>
        )}

        {/* File Upload */}
        {isDetectorReady && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
            <div className="flex items-center space-x-4">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="flex-1 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              <button
                onClick={resetAll}
                className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                Reset
              </button>
            </div>
          </div>
        )}

        {/* Model Selection */}
        {isDetectorReady && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">🤖 Model Selection</h2>
              <div className="flex space-x-2">
                <button
                  onClick={selectAllModels}
                  disabled={
                    availableModels.length === 0 ||
                    selectedModels.length === availableModels.length
                  }
                  className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-3 py-1 rounded text-sm transition-colors"
                >
                  Select All
                </button>
                <button
                  onClick={clearAllModels}
                  disabled={selectedModels.length === 0}
                  className="bg-gray-500 hover:bg-gray-600 disabled:bg-gray-300 text-white px-3 py-1 rounded text-sm transition-colors"
                >
                  Clear All
                </button>
              </div>
            </div>

            {isLoadingModels ? (
              <div className="text-center py-4">
                <p className="text-gray-600">Loading available models...</p>
              </div>
            ) : availableModels.length > 0 ? (
              <div>
                <p className="text-sm text-gray-600 mb-3">
                  Select models to use for classification. Multiple models will
                  run in parallel for comparison.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {availableModels.map((modelName) => (
                    <label
                      key={modelName}
                      className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(modelName)}
                        onChange={(e) =>
                          handleModelSelection(modelName, e.target.checked)
                        }
                        className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                      />
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">
                          {modelName}
                        </div>
                        <div className="text-sm text-gray-500">
                          TensorFlow/Keras Model
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
                {selectedModels.length > 0 && (
                  <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <p className="text-sm text-blue-800">
                      <span className="font-medium">
                        {selectedModels.length} model(s) selected:
                      </span>{" "}
                      {selectedModels.join(", ")}
                    </p>
                    <p className="text-xs text-blue-600 mt-1">
                      Each model will process the image independently and
                      results will be compared
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-4">
                <p className="text-gray-600">
                  No models available. Please ensure your backend is running and
                  models are loaded.
                </p>
                <button
                  onClick={loadAvailableModels}
                  disabled={isLoadingModels}
                  className="mt-2 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded text-sm transition-colors"
                >
                  Retry Loading Models
                </button>
              </div>
            )}
          </div>
        )}

        {/* Confidence Threshold Control */}
        {isDetectorReady && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">🎯 Confidence Filter</h2>
              <div className="text-sm text-gray-600">
                Current: {(confidenceThreshold * 100).toFixed(0)}% above
              </div>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={() => setConfidenceThreshold(0.9)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  confidenceThreshold === 0.9 
                    ? 'bg-red-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                90% above
              </button>
              <button
                onClick={() => setConfidenceThreshold(0.8)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  confidenceThreshold === 0.8 
                    ? 'bg-orange-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                80% above
              </button>
              <button
                onClick={() => setConfidenceThreshold(0.7)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  confidenceThreshold === 0.7 
                    ? 'bg-yellow-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                70% above
              </button>
              <button
                onClick={() => setConfidenceThreshold(0.5)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  confidenceThreshold === 0.5 
                    ? 'bg-green-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                50% above
              </button>
              <button
                onClick={() => setConfidenceThreshold(0.0)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  confidenceThreshold === 0.0 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Show all
              </button>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Original Image */}
        {uploadedImage && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Original Image</h2>
              <div className="flex space-x-3">
                <button
                  onClick={getImageInfo}
                  className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                >
                  📏 Image Info
                </button>
                <button
                  onClick={detectTrafficSigns}
                  disabled={isLoading}
                  className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                >
                  {isLoading ? "Detecting..." : "Detect Traffic Signs"}
                </button>
              </div>
            </div>
            <div className="text-center">
              <img
                ref={imageRef}
                src={uploadedImage}
                alt="Uploaded image"
                className="max-w-full max-h-96 mx-auto rounded-lg shadow-sm"
                crossOrigin="anonymous"
              />
              <p className="text-sm text-gray-600 text-center mt-2">
                Original size: {imageRef.current?.naturalWidth} × {imageRef.current?.naturalHeight}px
                {imageRef.current?.naturalWidth !== imageRef.current?.width && (
                  <span className="text-gray-500 ml-2">
                    (displayed: {imageRef.current?.width} × {imageRef.current?.height}px)
                  </span>
                )}
              </p>
            </div>
          </div>
        )}

        {/* Processing Steps */}
        {processingSteps && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center space-x-3">
                <h2 className="text-xl font-semibold">🔬 Processing Steps</h2>
                <button
                  onClick={() => setIsProcessingStepsCollapsed(!isProcessingStepsCollapsed)}
                  className="text-gray-500 hover:text-gray-700 transition-colors"
                >
                  {isProcessingStepsCollapsed ? (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                    </svg>
                  )}
                </button>
              </div>
              <div className="flex items-center space-x-4">
                {scaleFactor > 1 && (
                  <div className="bg-blue-50 px-3 py-1 rounded-full">
                    <span className="text-sm font-medium text-blue-700">
                      Upscaled {scaleFactor.toFixed(1)}x
                    </span>
                  </div>
                )}
                <span className="text-sm text-gray-500">
                  {processingSteps.length} steps
                </span>
              </div>
            </div>

            {!isProcessingStepsCollapsed && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {processingSteps.map((step, index) => (
                  <div key={index}>
                    <h3 className="font-medium mb-2 text-sm text-gray-700">
                      {index + 1}. {step.name}
                    </h3>
                    <img
                      src={step.image}
                      alt={step.name}
                      className="w-full h-48 object-contain rounded-lg border border-gray-200"
                    />
                  </div>
                ))}
              </div>
            )}

            {isProcessingStepsCollapsed && (
              <div className="text-center py-4">
                <p className="text-gray-500 text-sm">
                  Click to expand and view {processingSteps.length} processing steps
                </p>
              </div>
            )}
          </div>
        )}

        {/* Detection Results */}
        {detections.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">
              🎯 Detection Results ({detections.length} signs found)
            </h2>
            <div className="grid grid-cols-1 gap-4 mb-4">
              {detections.map((detection, index) => (
                <div
                  key={index}
                  className="bg-gray-50 p-4 rounded-lg border border-gray-200"
                >
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="font-semibold text-lg">Sign {index + 1}</h3>
                    <div className="text-right">
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm font-medium">
                        {(detection.confidence * 100).toFixed(1)}% detected
                      </span>
                    </div>
                  </div>

                  {/* AI Classification Results */}
                  {detection.classification ? (
                    <div className="space-y-3">
                      <div className="bg-green-50 p-3 rounded-lg border border-green-200">
                        <div className="flex justify-between items-center mb-2">
                          <h4 className="font-medium text-green-800">
                            🤖 Best AI Classification
                          </h4>
                          <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm font-medium">
                            {(
                              detection.classification.ai_confidence * 100
                            ).toFixed(1)}
                            % confident
                          </span>
                        </div>
                        <p className="text-green-900 font-semibold text-lg">
                          {detection.classification.class_name}
                        </p>

                        {/* Top 3 Predictions */}
                        {detection.classification.top3_predictions && (
                          <div className="mt-3">
                            <h5 className="text-sm font-medium text-green-700 mb-2">
                              Top 3 Predictions:
                            </h5>
                            <div className="space-y-1">
                              {detection.classification.top3_predictions.map(
                                (pred, predIndex) => (
                                  <div
                                    key={predIndex}
                                    className="flex justify-between items-center text-sm"
                                  >
                                    <span
                                      className={
                                        predIndex === 0 ? "font-semibold" : ""
                                      }
                                    >
                                      {predIndex + 1}. {pred.class_name}
                                    </span>
                                    <span
                                      className={`px-2 py-1 rounded-full text-xs ${
                                        predIndex === 0
                                          ? "bg-green-200 text-green-800"
                                          : "bg-gray-200 text-gray-700"
                                      }`}
                                    >
                                      {(pred.confidence * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                )
                              )}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Multiple Model Results */}
                      {detection.multiModelResults &&
                        detection.multiModelResults.length > 1 && (
                          <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                            <h4 className="font-medium text-blue-800 mb-3">
                              📊 All Model Results (
                              {detection.multiModelResults.length} models)
                            </h4>
                            <div className="space-y-2">
                              {detection.multiModelResults
                                .sort((a, b) => b.confidence - a.confidence)
                                .map((result, resultIndex) => (
                                  <div
                                    key={resultIndex}
                                    className="bg-white p-2 rounded border border-blue-200"
                                  >
                                    <div className="flex justify-between items-center">
                                      <div className="flex-1">
                                        <div className="font-medium text-sm">
                                          {result.model_name}
                                        </div>
                                        <div className="text-sm text-gray-600">
                                          {result.class_name}
                                        </div>
                                        <div className="text-xs text-gray-500">
                                          Input: {result.input_size} • Type:{" "}
                                          {result.model_type}
                                        </div>
                                      </div>
                                      <div className="text-right">
                                        <span
                                          className={`px-2 py-1 rounded-full text-xs font-medium ${
                                            resultIndex === 0
                                              ? "bg-blue-200 text-blue-800"
                                              : "bg-gray-200 text-gray-700"
                                          }`}
                                        >
                                          {(result.confidence * 100).toFixed(1)}
                                          %
                                        </span>
                                        {resultIndex === 0 && (
                                          <div className="text-xs text-blue-600 mt-1">
                                            Highest
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                ))}
                            </div>
                            <div className="mt-2 text-xs text-blue-600">
                              🔄 All models processed the same region
                              independently
                            </div>
                          </div>
                        )}

                      {/* Model Agreement Analysis */}
                      {detection.multiModelResults &&
                        detection.multiModelResults.length > 1 && (
                          <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
                            <h4 className="font-medium text-purple-800 mb-2">
                              📈 Model Agreement
                            </h4>
                            <div className="text-sm text-purple-700">
                              {(() => {
                                const predictions =
                                  detection.multiModelResults.map(
                                    (r) => r.class_name
                                  );
                                const uniquePredictions = [
                                  ...new Set(predictions),
                                ];
                                const agreement =
                                  (predictions.length -
                                    uniquePredictions.length +
                                    1) /
                                  predictions.length;

                                return (
                                  <div>
                                    <div className="flex justify-between items-center">
                                      <span>Agreement Score:</span>
                                      <span
                                        className={`font-medium ${
                                          agreement > 0.7
                                            ? "text-green-600"
                                            : agreement > 0.5
                                            ? "text-yellow-600"
                                            : "text-red-600"
                                        }`}
                                      >
                                        {(agreement * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                    <div className="text-xs mt-1">
                                      {uniquePredictions.length === 1
                                        ? "All models agree on the prediction"
                                        : `${uniquePredictions.length} different predictions across models`}
                                    </div>
                                  </div>
                                );
                              })()}
                            </div>
                          </div>
                        )}
                    </div>
                  ) : (
                    <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                      <p className="text-yellow-800 text-sm">
                        ⚠️ AI classification unavailable (backend not connected
                        or no models selected)
                      </p>
                    </div>
                  )}

                  {/* Technical Details */}
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-600">
                      <strong>Position:</strong> ({detection.x}, {detection.y})
                      •<strong> Size:</strong> {detection.width}×
                      {detection.height}px
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Detected Image with Highlights */}
        {detectedImageUrl && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">
              Image with Detections
            </h2>
            <p className="text-sm text-gray-600 text-center mb-2">
              Processing dimensions: {imageRef.current?.naturalWidth} × {imageRef.current?.naturalHeight}px
            </p>
            <div className="text-center">
              <img
                src={detectedImageUrl}
                alt="Image with detected traffic signs highlighted"
                className="max-w-full max-h-96 mx-auto rounded-lg shadow-sm"
              />
            </div>
          </div>
        )}

        {/* Cropped Traffic Sign */}
        {croppedImageUrl && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">
                Cropped Traffic Sign (Best Detection)
              </h2>
              <button
                onClick={downloadCroppedImage}
                className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                Download Cropped Image
              </button>
            </div>
            <div className="text-center">
              <img
                src={croppedImageUrl}
                alt="Cropped traffic sign"
                className="max-w-full max-h-64 mx-auto rounded-lg shadow-sm border-2 border-gray-200"
              />
            </div>
            <p className="text-sm text-gray-600 text-center mt-2">
              This cropped region can be sent to your next processing step or
              API.
            </p>
          </div>
        )}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-xl">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <p className="text-lg font-medium">Processing...</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
