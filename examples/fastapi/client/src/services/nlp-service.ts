/**
 * NLP Service for communicating with the FastAPI C++ backend.
 * Refactored for React + TypeScript with robust error handling and SSE management.
 */

export interface NLPRequest {
  text: string;
  plugin?: string;
  options?: Record<string, any>;
  streaming?: boolean;
}

export interface MarkovRequest {
  seed: string;
  length?: number;
  model?: string;
  temperature?: number;
  session_id?: string;
}

export interface NLPResponse {
  result: string;
  task_id?: string;
  status: string;
}

export interface StreamChunk {
  chunk: string;
  is_final: boolean;
  task_id?: string;
  error?: string;
}

export interface HealthStatus {
  status: string;
  engine_ready: boolean;
}

export type StreamCallback = (data: StreamChunk) => void;
export type ErrorCallback = (error: any) => void;

class NLPService {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  /**
   * Check if the NLP engine is healthy and the C++ core is initialized.
   */
  async checkHealth(): Promise<HealthStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error("Health check failed");
      const data = await response.json();
      return {
        status: data.status || "unknown",
        engine_ready: !!data.engine_ready,
      };
    } catch (error) {
      console.error("NLP Service Health Check Error:", error);
      return { status: "unreachable", engine_ready: false };
    }
  }

  /**
   * Synchronous processing for immediate results.
   * Maps to the /process endpoint with streaming=false.
   */
  async processSync(request: NLPRequest): Promise<NLPResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...request,
          streaming: false,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Processing failed");
      }

      return await response.json();
    } catch (error) {
      console.error("NLP Sync Process Error:", error);
      throw error;
    }
  }

  /**
   * Initiates an asynchronous processing task.
   * Returns a task_id that can be used for status polling or streaming.
   */
  async submitAsyncTask(request: NLPRequest): Promise<{ task_id: string; status: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/async-process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Async submission failed");
      }

      return await response.json();
    } catch (error) {
      console.error("NLP Async Submit Error:", error);
      throw error;
    }
  }

  /**
   * Polls for the status of a specific task.
   */
  async getTaskStatus(taskId: string): Promise<{ task_id: string; status: string }> {
    const response = await fetch(`${this.baseUrl}/tasks/${taskId}`);
    if (!response.ok) throw new Error("Failed to fetch task status");
    return await response.json();
  }

  /**
   * Streaming interface using Server-Sent Events (SSE).
   * In this architecture, we first submit the task to get a task_id,
   * then connect to the /stream/{task_id} endpoint.
   */
  async streamNLP(request: NLPRequest, onChunk: StreamCallback, onError?: ErrorCallback): Promise<() => void> {
    let eventSource: EventSource | null = null;
    let isCancelled = false;

    const cleanup = () => {
      isCancelled = true;
      if (eventSource) {
        eventSource.close();
      }
    };

    try {
      // 1. Submit the task as an async process
      const { task_id } = await this.submitAsyncTask({ ...request, streaming: true });

      if (isCancelled) return cleanup;

      // 2. Connect to the SSE stream using the task_id
      // We pass the text as a query parameter so the engine can process it immediately
      const url = new URL(`${this.baseUrl}/stream/${task_id}`);
      url.searchParams.append("text", request.text);

      // Append additional options as query parameters for the stream
      if (request.options) {
        Object.entries(request.options).forEach(([key, value]) => {
          url.searchParams.append(key, String(value));
        });
      }

      eventSource = new EventSource(url.toString());

      eventSource.onmessage = (event) => {
        try {
          const data: StreamChunk = JSON.parse(event.data);

          if (data.error) {
            if (onError) onError(data.error);
            eventSource?.close();
            return;
          }

          onChunk(data);

          if (data.is_final) {
            eventSource?.close();
          }
        } catch (err) {
          console.error("Failed to parse SSE message:", err);
          if (onError) onError(err);
        }
      };

      eventSource.onerror = (err) => {
        // SSE error events don't provide much detail in browser
        console.error("SSE Connection Error for task:", task_id);
        if (onError) onError(new Error("Stream connection lost"));
        eventSource?.close();
      };
    } catch (error) {
      if (onError) onError(error);
    }

    return cleanup;
  }

  /**
   * Generates text using a Markov model addon.
   */
  async generateMarkov(request: MarkovRequest): Promise<{ output: string; status: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: request.seed,
          plugin: request.model || "markov_generator",
          session_id: request.session_id,
          options: {
            length: request.length || 100,
            temperature: request.temperature || 1.0,
          },
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Generation failed");
      }

      return await response.json();
    } catch (error) {
      console.error("NLP Generation Error:", error);
      throw error;
    }
  }

  /**
   * Fetches the list of available Markov models from the backend.
   */
  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      if (!response.ok) return ["markov_generator"];
      const data = await response.json();
      return data.available_models || ["markov_generator"];
    } catch (error) {
      console.error("Failed to fetch models:", error);
      return ["markov_generator"];
    }
  }
}

// Export as a singleton for use throughout the app
export const nlpService = new NLPService();
