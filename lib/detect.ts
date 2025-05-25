declare global {
  interface Window {
    cv: any; // eslint-disable-line @typescript-eslint/no-explicit-any
  }
}

export interface ModelResult {
  model_name: string;
  predicted_class: number;
  class_name: string;
  confidence: number;
  model_type: string;
  input_size: string;
  top3_predictions: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
  }>;
}

export interface DetectedSign {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  // AI Classification results - can be from multiple models
  classification?: {
    predicted_class: number;
    class_name: string;
    ai_confidence: number;
    top3_predictions: Array<{
      class_id: number;
      class_name: string;
      confidence: number;
    }>;
  };
  // Multiple model results
  multiModelResults?: ModelResult[];
}

export interface ProcessingSteps {
  name: string;
  image: string;
}

export interface DetectionResult {
  detections: DetectedSign[];
  processingSteps: ProcessingSteps[];
  scaleFactor: number;
}

export class TrafficSignDetector {
  private cv: any; // eslint-disable-line @typescript-eslint/no-explicit-any
  private isReady: boolean = false;
  private readonly minImageSize = 500; // Minimum size for the longest dimension
  private readonly API_BASE =
    process.env.NEXT_PUBLIC_API_URL ||
    "https://ahmeetkok-traffic-sign-classifier.hf.space";

  constructor() {
    this.cv = null;
  }

  async initialize(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (typeof window === "undefined") {
        reject(new Error("OpenCV.js can only be used in browser environment"));
        return;
      }

      // Check if opencv.js is already loaded
      if (window.cv && window.cv.Mat) {
        this.cv = window.cv;
        this.isReady = true;
        resolve();
        return;
      }

      // Load opencv.js
      const script = document.createElement("script");
      script.src = "/opencv.js";
      script.onload = () => {
        // Wait for cv to be ready
        const checkCV = () => {
          if (window.cv && window.cv.Mat) {
            this.cv = window.cv;
            this.isReady = true;
            resolve();
          } else {
            setTimeout(checkCV, 100);
          }
        };
        checkCV();
      };
      script.onerror = () => reject(new Error("Failed to load OpenCV.js"));
      document.head.appendChild(script);
    });
  }

  private matToCanvas(mat: unknown): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    this.cv.imshow(canvas, mat);
    return canvas;
  }

  private upscaleImage(imageElement: HTMLImageElement): {
    canvas: HTMLCanvasElement;
    scaleFactor: number;
  } {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Could not get canvas context");
    }

    const originalWidth = imageElement.naturalWidth;
    const originalHeight = imageElement.naturalHeight;
    // const maxDimension = Math.max(originalWidth, originalHeight);

    // Calculate scale factor - only upscale if image is smaller than minimum size
    const scaleFactor = 1;
    /* if (maxDimension < this.minImageSize) {
      scaleFactor = this.minImageSize / maxDimension;
    } */

    const newWidth = Math.round(originalWidth * scaleFactor);
    const newHeight = Math.round(originalHeight * scaleFactor);

    canvas.width = newWidth;
    canvas.height = newHeight;

    // Use high-quality scaling
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctx.drawImage(imageElement, 0, 0, newWidth, newHeight);

    return { canvas, scaleFactor };
  }

  async detectTrafficSignsWithSteps(
    imageElement: HTMLImageElement
  ): Promise<DetectionResult> {
    if (!this.isReady || !this.cv) {
      throw new Error("OpenCV.js is not initialized");
    }

    // Step 1: Upscale if needed
    const { canvas: upscaledCanvas, scaleFactor } =
      this.upscaleImage(imageElement);
    const upscaledImageUrl =
      scaleFactor > 1 ? upscaledCanvas.toDataURL() : undefined;

    // Use the upscaled canvas for processing
    const workingCanvas = upscaledCanvas;

    // Convert canvas to OpenCV Mat
    const src = this.cv.imread(workingCanvas);
    const gray = new this.cv.Mat();
    const contrast = new this.cv.Mat();
    const bilateral = new this.cv.Mat();
    const blurred = new this.cv.Mat();
    const morphed = new this.cv.Mat();
    const edges = new this.cv.Mat();
    const contours = new this.cv.MatVector();
    const hierarchy = new this.cv.Mat();
    const contourDisplay = src.clone();
    let kernel: any = null; // eslint-disable-line @typescript-eslint/no-explicit-any

    try {
      // Step 2: Original image (after upscaling if applied)
      const originalCanvas = this.matToCanvas(src);

      // Step 3: Convert to grayscale
      this.cv.cvtColor(src, gray, this.cv.COLOR_RGBA2GRAY);
      const grayscaleCanvas = this.matToCanvas(gray);

      // Step 4: Increase contrast of grayscaled image
      // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
      const clahe = new this.cv.CLAHE(2.0, new this.cv.Size(8, 8));
      clahe.apply(gray, contrast);
      const contrastCanvas = this.matToCanvas(contrast);
      clahe.delete();

      // Step 5: Advanced noise reduction with bilateral filter
      this.cv.bilateralFilter(contrast, bilateral, 9, 75, 75);
      const bilateralCanvas = this.matToCanvas(bilateral);

      // Step 6: Apply stronger Gaussian blur to reduce noise further
      // Increased kernel sizes for better noise reduction
      const kernelSize = scaleFactor > 1 ? 11 : 9; // Increased from 7:5
      this.cv.GaussianBlur(
        bilateral,
        blurred,
        new this.cv.Size(kernelSize, kernelSize),
        0
      );
      const blurredCanvas = this.matToCanvas(blurred);

      // Step 7: Morphological operations to reduce noise and fill gaps
      kernel = this.cv.getStructuringElement(
        this.cv.MORPH_ELLIPSE,
        new this.cv.Size(3, 3)
      );
      // Opening operation (erosion followed by dilation) to remove noise
      this.cv.morphologyEx(blurred, morphed, this.cv.MORPH_OPEN, kernel);
      const morphedCanvas = this.matToCanvas(morphed);

      // Step 8: Edge detection with adjusted thresholds for better triangle detection
      const lowThreshold = scaleFactor > 1 ? 25 : 40; // Lowered for better triangle detection
      const highThreshold = scaleFactor > 1 ? 80 : 120; // Lowered for better triangle detection
      this.cv.Canny(morphed, edges, lowThreshold, highThreshold);
      const edgesCanvas = this.matToCanvas(edges);

      // Step 9: Find contours
      this.cv.findContours(
        edges,
        contours,
        hierarchy,
        this.cv.RETR_EXTERNAL,
        this.cv.CHAIN_APPROX_SIMPLE
      );

      // Draw all contours for visualization
      for (let i = 0; i < contours.size(); i++) {
        const color = new this.cv.Scalar(0, 255, 0, 255); // Green color
        this.cv.drawContours(
          contourDisplay,
          contours,
          i,
          color,
          Math.max(2, Math.round(2 * scaleFactor))
        );
      }
      const contoursCanvas = this.matToCanvas(contourDisplay);

      const detectedSigns: DetectedSign[] = [];

      // Analyze contours for potential traffic signs with scaled thresholds
      /*       const minArea = 100 * (scaleFactor * scaleFactor);
      const maxArea = 50000 * (scaleFactor * scaleFactor); */

      for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = this.cv.contourArea(contour);

        // Filter by area (traffic signs should have reasonable size)
        const rect = this.cv.boundingRect(contour);
        const aspectRatio = rect.width / rect.height;

        // Focus on square-like traffic signs (aspect ratio 0.8 to 1.2)
        // This catches most standard traffic signs while filtering out very elongated shapes
        if (aspectRatio > 0.8 && aspectRatio < 1.2) {
          // Additional checks for traffic sign characteristics
          const perimeter = this.cv.arcLength(contour, true);
          const approx = new this.cv.Mat();
          this.cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);

          // Get number of vertices for shape analysis
          const vertices = approx.rows;

          // Calculate confidence based on shape characteristics
          let confidence = 0.4; // Lower base confidence to catch more shapes

          // Prefer larger areas (adjust for scale)
          confidence += Math.min(
            area / (8000 * scaleFactor * scaleFactor), // Lowered threshold
            0.3
          );

          // Shape-specific confidence boosts
          if (vertices >= 3 && vertices <= 8) {
            // Triangle detection (3-4 vertices due to approximation)
            if (vertices === 3 || vertices === 4) {
              confidence += 0.2; // Boost for triangular shapes

              // Additional boost for inverted triangles (taller than wide)
              if (aspectRatio < 0.9) {
                confidence += 0.1;
              }
            }

            // Square/rectangular signs (4-6 vertices)
            else if (vertices >= 4 && vertices <= 6) {
              // Prefer square-like shapes
              confidence += (1 - Math.abs(aspectRatio - 1)) * 0.2;
            }

            // Circular/octagonal signs (6-8 vertices)
            else if (vertices >= 6 && vertices <= 8) {
              confidence += 0.15;
              // Prefer circular shapes (aspect ratio close to 1)
              confidence += (1 - Math.abs(aspectRatio - 1)) * 0.15;
            }
          }

          // Compactness check (perimeter² / area) - traffic signs are usually compact
          const compactness = (perimeter * perimeter) / (4 * Math.PI * area);
          if (compactness < 3.0) {
            // More lenient for triangles
            confidence += 0.1;
          }

          // Scale back coordinates to original image size
          if (confidence > 0.7) { // Lowered from 0.75 to 0.3 to allow more detections through
            detectedSigns.push({
              x: Math.round(rect.x / scaleFactor),
              y: Math.round(rect.y / scaleFactor),
              width: Math.round(rect.width / scaleFactor),
              height: Math.round(rect.height / scaleFactor),
              confidence: Math.min(confidence, 1.0),
            });
          }

          approx.delete();
        }

        contour.delete();
      }

      // Sort by confidence and return top detections
      const sortedDetections = detectedSigns
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 5); // Return top 5 detections

      // Additional triangle detection using Hough lines
      const triangleDetections = this.detectTrianglesWithHoughLines(
        edges,
        scaleFactor
      );

      // Merge triangle detections with main detections, avoiding duplicates
      const allDetections = [...sortedDetections];
      triangleDetections.forEach((triDetection) => {
        // Check if this triangle detection overlaps significantly with existing detections
        const hasOverlap = allDetections.some((existing) => {
          const overlapX = Math.max(
            0,
            Math.min(
              existing.x + existing.width,
              triDetection.x + triDetection.width
            ) - Math.max(existing.x, triDetection.x)
          );
          const overlapY = Math.max(
            0,
            Math.min(
              existing.y + existing.height,
              triDetection.y + triDetection.height
            ) - Math.max(existing.y, triDetection.y)
          );
          const overlapArea = overlapX * overlapY;
          const existingArea = existing.width * existing.height;
          const triArea = triDetection.width * triDetection.height;
          const overlapRatio = overlapArea / Math.min(existingArea, triArea);
          return overlapRatio > 0.3; // 30% overlap threshold
        });

        if (!hasOverlap) {
          allDetections.push(triDetection);
        }
      });

      // Re-sort and limit final detections
      const finalDetections = allDetections
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 8); // Increased to 8 to accommodate more triangle detections

      const processingSteps: ProcessingSteps[] = [];

      // Add upscaled step if upscaling was applied
      if (upscaledImageUrl) {
        processingSteps.push({
          name: "Upscaled",
          image: upscaledImageUrl,
        });
      }
      processingSteps.push({
        name: "Original",
        image: originalCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Grayscale",
        image: grayscaleCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Contrast",
        image: contrastCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Contrast Enhanced",
        image: contrastCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Bilateral Filter",
        image: bilateralCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Gaussian Blur",
        image: blurredCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Morphological Filter",
        image: morphedCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Edge Detection",
        image: edgesCanvas.toDataURL(),
      });
      processingSteps.push({
        name: "Contours",
        image: contoursCanvas.toDataURL(),
      });
      return {
        detections: finalDetections,
        processingSteps,
        scaleFactor,
      };
    } finally {
      // Clean up OpenCV objects
      src.delete();
      gray.delete();
      contrast.delete();
      bilateral.delete();
      blurred.delete();
      morphed.delete();
      if (kernel) kernel.delete();
      edges.delete();
      contours.delete();
      hierarchy.delete();
      contourDisplay.delete();
    }
  }

  // Keep the original method for backward compatibility
  async detectTrafficSigns(
    imageElement: HTMLImageElement
  ): Promise<DetectedSign[]> {
    const result = await this.detectTrafficSignsWithSteps(imageElement);
    return result.detections;
  }

  cropDetectedRegion(
    imageElement: HTMLImageElement,
    detection: DetectedSign
  ): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Could not get canvas context");
    }

    // Add some padding around the detected region
    const padding = 1;
    const x = Math.max(0, detection.x - padding);
    const y = Math.max(0, detection.y - padding);
    const width = Math.min(
      imageElement.naturalWidth - x,
      detection.width + padding * 2
    );
    const height = Math.min(
      imageElement.naturalHeight - y,
      detection.height + padding * 2
    );

    canvas.width = width;
    canvas.height = height;

    // Draw the cropped region
    ctx.drawImage(
      imageElement,
      x,
      y,
      width,
      height, // source rectangle
      0,
      0,
      width,
      height // destination rectangle
    );

    return canvas;
  }

  drawDetections(
    imageElement: HTMLImageElement,
    detections: DetectedSign[]
  ): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Could not get canvas context");
    }

    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;

    // Draw the original image
    ctx.drawImage(imageElement, 0, 0);

    // Draw detection rectangles
    detections.forEach((detection) => {
      ctx.strokeStyle = "#00ff00";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        detection.x,
        detection.y,
        detection.width,
        detection.height
      );

      // Draw confidence label
      ctx.fillStyle = "#00ff00";
      ctx.font = "16px Arial";
      ctx.fillText(
        `${(detection.confidence * 100).toFixed(1)}%`,
        detection.x,
        detection.y - 5
      );
    });

    return canvas;
  }

  async classifyDetectedRegions(
    imageElement: HTMLImageElement,
    detections: DetectedSign[]
  ): Promise<DetectedSign[]> {
    if (detections.length === 0) return detections;

    try {
      // Convert image element to blob
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get canvas context");

      canvas.width = imageElement.naturalWidth;
      canvas.height = imageElement.naturalHeight;
      ctx.drawImage(imageElement, 0, 0);

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob(resolve as BlobCallback, "image/jpeg", 0.8);
      });

      if (!blob) throw new Error("Could not convert image to blob");

      const formData = new FormData();
      formData.append("file", blob, "image.jpg");
      formData.append(
        "regions",
        JSON.stringify(
          detections.map((d) => ({
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
          }))
        )
      );
      console.log(formData.get("regions"));
      console.log(formData.get("file"));
      // Call backend API
      const response = await fetch(`${this.API_BASE}/classify-regions`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        console.warn("Backend classification failed:", response.statusText);
        return detections; // Return original detections if backend fails
      }

      const result = await response.json();

      if (result.success && result.results) {
        // Merge classification results with detections
        const classifiedDetections = detections.map((detection, index) => {
          const classification = result.results.find(
            (r: { region_id: number }) => r.region_id === index
          );
          if (classification?.prediction) {
            return {
              ...detection,
              classification: {
                predicted_class: classification.prediction.predicted_class,
                class_name: classification.prediction.class_name,
                ai_confidence: classification.prediction.confidence,
                top3_predictions: classification.prediction.top3_predictions,
              },
            };
          }
          return detection;
        });

        console.log(
          `✅ Successfully classified ${result.results.length} regions`
        );
        return classifiedDetections;
      }

      return detections;
    } catch (error) {
      console.warn("Error during classification:", error);
      return detections; // Return original detections if classification fails
    }
  }

  async classifyEntireImage(
    imageElement: HTMLImageElement
  ): Promise<DetectedSign[]> {
    try {
      console.log("🔄 No regions detected, trying to classify entire image...");

      // Convert image element to blob
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get canvas context");

      canvas.width = imageElement.naturalWidth;
      canvas.height = imageElement.naturalHeight;
      ctx.drawImage(imageElement, 0, 0);

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob(resolve as BlobCallback, "image/jpeg", 0.8);
      });

      if (!blob) throw new Error("Could not convert image to blob");

      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      // Call single image classification endpoint
      const response = await fetch(`${this.API_BASE}/classify`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        console.warn("Backend classification failed:", response.statusText);
        return [];
      }

      const result = await response.json();

      if (result.success && result.result) {
        // Create a detection covering the entire image
        const fullImageDetection: DetectedSign = {
          x: 0,
          y: 0,
          width: imageElement.naturalWidth,
          height: imageElement.naturalHeight,
          confidence: 0.5, // Base confidence for full image classification
          classification: {
            predicted_class: result.result.predicted_class,
            class_name: result.result.class_name,
            ai_confidence: result.result.confidence,
            top3_predictions: result.result.top3_predictions,
          },
        };

        console.log(`✅ Full image classified as: ${result.result.class_name}`);
        return [fullImageDetection];
      }

      return [];
    } catch (error) {
      console.warn("Error during full image classification:", error);
      return [];
    }
  }

  async getAvailableModels(): Promise<string[]> {
    try {
      console.log("🔍 Fetching available models...");

      // Use Hugging Face Spaces API endpoint
      const response = await fetch(`${this.API_BASE}/models`, {
        method: "GET",
      });

      if (!response.ok) {
        console.warn("Failed to fetch models:", response.statusText);
        return [];
      }

      const result = await response.json();

      if (result.success && result.available_models) {
        console.log(
          `✅ Found ${result.available_models.length} models:`,
          result.available_models
        );
        return result.available_models;
      }

      return [];
    } catch (error) {
      console.warn("Error fetching available models:", error);
      return [];
    }
  }

  async classifyWithMultipleModels(
    imageElement: HTMLImageElement,
    detections: DetectedSign[],
    selectedModels: string[]
  ): Promise<DetectedSign[]> {
    if (selectedModels.length === 0) {
      console.warn("No models selected for classification");
      return detections;
    }

    console.log(
      `🤖 Classifying with ${selectedModels.length} models:`,
      selectedModels
    );

    try {
      // Convert image element to blob
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get canvas context");

      canvas.width = imageElement.naturalWidth;
      canvas.height = imageElement.naturalHeight;
      ctx.drawImage(imageElement, 0, 0);

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob(resolve as BlobCallback, "image/jpeg", 0.8);
      });

      if (!blob) throw new Error("Could not convert image to blob");

      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      // Prepare regions data
      const regions = detections.map((detection) => ({
        x: detection.x,
        y: detection.y,
        width: detection.width,
        height: detection.height,
      }));

      formData.append("regions", JSON.stringify(regions));

      // Send parallel requests for each selected model
      const modelPromises = selectedModels.map(async (modelName) => {
        const modelFormData = new FormData();
        modelFormData.append("file", blob, "image.jpg");
        modelFormData.append("regions", JSON.stringify(regions));
        modelFormData.append("model_name", modelName);
        console.log(modelFormData.get("model_name"));
        const response = await fetch(`${this.API_BASE}/classify-regions`, {
          method: "POST",
          body: modelFormData,
        });

        if (!response.ok) {
          console.warn(
            `Backend classification failed for ${modelName}:`,
            response.statusText
          );
          return { modelName, success: false, error: response.statusText };
        }

        const result = await response.json();

        if (result.success) {
          return { modelName, success: true, results: result.results };
        } else {
          return { modelName, success: false, error: "Classification failed" };
        }
      });

      // Wait for all model results
      const modelResults = await Promise.all(modelPromises);

      // Process results and combine them
      const updatedDetections = detections.map((detection, detectionIndex) => {
        const multiModelResults: ModelResult[] = [];

        modelResults.forEach((modelResult) => {
          if (modelResult.success && modelResult.results) {
            const regionResult = modelResult.results[detectionIndex];
            if (regionResult && regionResult.prediction) {
              multiModelResults.push({
                ...regionResult.prediction,
                model_name: modelResult.modelName,
              });
            }
          }
        });

        // Find the best result (highest confidence) for backward compatibility
        let bestResult = null;
        if (multiModelResults.length > 0) {
          bestResult = multiModelResults.reduce((best, current) =>
            current.confidence > best.confidence ? current : best
          );
        }

        return {
          ...detection,
          // Keep single result for backward compatibility
          classification: bestResult
            ? {
                predicted_class: bestResult.predicted_class,
                class_name: bestResult.class_name,
                ai_confidence: bestResult.confidence,
                top3_predictions: bestResult.top3_predictions,
              }
            : undefined,
          // Add all model results
          multiModelResults: multiModelResults,
        };
      });

      console.log(
        `✅ Multi-model classification complete! Got results from ${
          modelResults.filter((r) => r.success).length
        }/${selectedModels.length} models`
      );
      return updatedDetections;
    } catch (error) {
      console.warn("Error during multi-model classification:", error);
      return detections;
    }
  }

  async classifyEntireImageWithMultipleModels(
    imageElement: HTMLImageElement,
    selectedModels: string[]
  ): Promise<DetectedSign[]> {
    if (selectedModels.length === 0) {
      console.warn("No models selected for full image classification");
      return [];
    }

    try {
      console.log(
        "🔄 No regions detected, trying full image classification with multiple models..."
      );

      // Convert image element to blob
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get canvas context");

      canvas.width = imageElement.naturalWidth;
      canvas.height = imageElement.naturalHeight;
      ctx.drawImage(imageElement, 0, 0);

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob(resolve as BlobCallback, "image/jpeg", 0.8);
      });

      if (!blob) throw new Error("Could not convert image to blob");

      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      // Send parallel requests for each selected model
      const modelPromises = selectedModels.map(async (modelName) => {
        const modelFormData = new FormData();
        modelFormData.append("file", blob, "image.jpg");
        modelFormData.append("model_name", modelName);

        const response = await fetch(`${this.API_BASE}/classify`, {
          method: "POST",
          body: modelFormData,
        });

        if (!response.ok) {
          console.warn(
            `Backend classification failed for ${modelName}:`,
            response.statusText
          );
          return { modelName, success: false, error: response.statusText };
        }

        const result = await response.json();

        if (result.success) {
          return { modelName, success: true, result: result.result };
        } else {
          return { modelName, success: false, error: "Classification failed" };
        }
      });

      // Wait for all model results
      const modelResults = await Promise.all(modelPromises);

      // Combine results
      const multiModelResults: ModelResult[] = [];
      modelResults.forEach((modelResult) => {
        if (modelResult.success && modelResult.result) {
          multiModelResults.push({
            ...modelResult.result,
            model_name: modelResult.modelName,
          });
        }
      });

      if (multiModelResults.length > 0) {
        // Find the best result (highest confidence) for the main classification
        const bestResult = multiModelResults.reduce((best, current) =>
          current.confidence > best.confidence ? current : best
        );

        const fullImageDetection: DetectedSign = {
          x: 0,
          y: 0,
          width: imageElement.naturalWidth,
          height: imageElement.naturalHeight,
          confidence: 0.5, // Base confidence for full image classification
          classification: {
            predicted_class: bestResult.predicted_class,
            class_name: bestResult.class_name,
            ai_confidence: bestResult.confidence,
            top3_predictions: bestResult.top3_predictions,
          },
          multiModelResults: multiModelResults,
        };

        console.log(
          `✅ Full image classified with ${multiModelResults.length} models. Best: ${bestResult.class_name}`
        );
        return [fullImageDetection];
      }

      return [];
    } catch (error) {
      console.warn(
        "Error during multi-model full image classification:",
        error
      );
      return [];
    }
  }

  private detectTrianglesWithHoughLines(
    edges: unknown,
    scaleFactor: number
  ): DetectedSign[] {
    const triangleDetections: DetectedSign[] = [];

    try {
      // Use HoughLinesP to detect line segments
      const lines = new this.cv.Mat();
      this.cv.HoughLinesP(
        edges,
        lines,
        1, // rho resolution
        Math.PI / 180, // theta resolution (1 degree)
        30, // threshold - minimum votes
        20 * scaleFactor, // minLineLength
        10 * scaleFactor // maxLineGap
      );

      // Group lines into potential triangles
      const lineSegments = [];
      for (let i = 0; i < lines.rows; i++) {
        const x1 = lines.data32S[i * 4];
        const y1 = lines.data32S[i * 4 + 1];
        const x2 = lines.data32S[i * 4 + 2];
        const y2 = lines.data32S[i * 4 + 3];

        const length = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;

        lineSegments.push({
          x1,
          y1,
          x2,
          y2,
          length,
          angle,
          midX: (x1 + x2) / 2,
          midY: (y1 + y2) / 2,
        });
      }

      // Look for triangular patterns
      for (let i = 0; i < lineSegments.length; i++) {
        for (let j = i + 1; j < lineSegments.length; j++) {
          for (let k = j + 1; k < lineSegments.length; k++) {
            const line1 = lineSegments[i];
            const line2 = lineSegments[j];
            const line3 = lineSegments[k];

            // Check if lines could form a triangle
            const angleDiff1 = Math.abs(line1.angle - line2.angle);
            const angleDiff2 = Math.abs(line2.angle - line3.angle);
            const angleDiff3 = Math.abs(line3.angle - line1.angle);

            // Lines should have different angles (not parallel)
            if (angleDiff1 > 20 && angleDiff2 > 20 && angleDiff3 > 20) {
              // Check if lines are close to each other (potential triangle)
              const maxDistance = 50 * scaleFactor;
              const dist12 = Math.sqrt(
                (line1.midX - line2.midX) ** 2 + (line1.midY - line2.midY) ** 2
              );
              const dist23 = Math.sqrt(
                (line2.midX - line3.midX) ** 2 + (line2.midY - line3.midY) ** 2
              );
              const dist31 = Math.sqrt(
                (line3.midX - line1.midX) ** 2 + (line3.midY - line1.midY) ** 2
              );

              if (
                dist12 < maxDistance &&
                dist23 < maxDistance &&
                dist31 < maxDistance
              ) {
                // Calculate bounding box of the potential triangle
                const minX = Math.min(
                  line1.x1,
                  line1.x2,
                  line2.x1,
                  line2.x2,
                  line3.x1,
                  line3.x2
                );
                const maxX = Math.max(
                  line1.x1,
                  line1.x2,
                  line2.x1,
                  line2.x2,
                  line3.x1,
                  line3.x2
                );
                const minY = Math.min(
                  line1.y1,
                  line1.y2,
                  line2.y1,
                  line2.y2,
                  line3.y1,
                  line3.y2
                );
                const maxY = Math.max(
                  line1.y1,
                  line1.y2,
                  line2.y1,
                  line2.y2,
                  line3.y1,
                  line3.y2
                );

                const width = maxX - minX;
                const height = maxY - minY;
                const area = width * height;

                // Filter by reasonable size
                if (
                  area > 500 * scaleFactor * scaleFactor &&
                  area < 50000 * scaleFactor * scaleFactor
                ) {
                  const aspectRatio = width / height;

                  // Calculate confidence for triangle detection
                  let confidence = 0.3; // Base confidence for Hough-detected triangles

                  // Boost for inverted triangles (taller than wide)
                  if (aspectRatio < 0.9) {
                    confidence += 0.2;
                  }

                  // Boost for reasonable triangle proportions
                  if (aspectRatio > 0.5 && aspectRatio < 1.5) {
                    confidence += 0.15;
                  }

                  // Scale back to original image coordinates
                  triangleDetections.push({
                    x: Math.round(minX / scaleFactor),
                    y: Math.round(minY / scaleFactor),
                    width: Math.round(width / scaleFactor),
                    height: Math.round(height / scaleFactor),
                    confidence: Math.min(confidence, 1.0),
                  });
                }
              }
            }
          }
        }
      }

      lines.delete();
    } catch (error) {
      console.warn("Error in triangle detection:", error);
    }

    return triangleDetections;
  }
}
