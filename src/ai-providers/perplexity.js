/**
 * src/ai-providers/perplexity.js
 *
 * Implementation for interacting with Perplexity models
 * using the Vercel AI SDK.
 */
import { createPerplexity } from '@ai-sdk/perplexity';
import { generateText, streamText, generateObject } from 'ai';
import { log } from '../../scripts/modules/utils.js';

// --- Client Instantiation ---
function getClient(apiKey, baseUrl) {
	if (!apiKey) {
		throw new Error('Perplexity API key is required.');
	}
	return createPerplexity({
		apiKey: apiKey,
		...(baseUrl && { baseURL: baseUrl })
	});
}

// --- Standardized Service Function Implementations ---

/**
 * Generates text using a Perplexity model.
 *
 * @param {object} params - Parameters for the text generation.
 * @param {string} params.apiKey - The Perplexity API key.
 * @param {string} params.modelId - The specific Perplexity model ID.
 * @param {Array<object>} params.messages - The messages array.
 * @param {number} [params.temperature] - Temperature for generation.
 * @param {string} [params.baseUrl] - Base URL for the Perplexity API.
 * @param {number} [params.contextWindowTokens] - Optional: Max context tokens for the model.
 * @param {number} [params.maxOutputTokens] - Optional: Max tokens for the response.
 * @returns {Promise<object>} The generated text content and usage.
 * @throws {Error} If the API call fails.
 */
export async function generatePerplexityText({
	apiKey,
	modelId,
	messages,
	temperature,
	baseUrl,
	contextWindowTokens,
	maxOutputTokens
}) {
	log('debug', `Generating Perplexity text with model: ${modelId}`);
	try {
		const client = getClient(apiKey, baseUrl);
		const result = await generateText({
			model: client(modelId),
			messages: messages,
			maxTokens: maxOutputTokens,
			temperature: temperature
		});
		log(
			'debug',
			`Perplexity generateText result received. Tokens: ${result.usage.completionTokens}/${result.usage.promptTokens}`
		);

		return {
			text: result.text,
			usage: {
				inputTokens: result.usage.promptTokens,
				outputTokens: result.usage.completionTokens
			}
		};
	} catch (error) {
		log('error', `Perplexity generateText failed: ${error.message}`);
		throw error;
	}
}

/**
 * Streams text using a Perplexity model.
 *
 * @param {object} params - Parameters for the text streaming.
 * @param {string} params.apiKey - The Perplexity API key.
 * @param {string} params.modelId - The specific Perplexity model ID.
 * @param {Array<object>} params.messages - The messages array.
 * @param {number} [params.temperature] - Temperature for generation.
 * @param {string} [params.baseUrl] - Base URL for the Perplexity API.
 * @param {number} [params.contextWindowTokens] - Optional: Max context tokens for the model.
 * @param {number} [params.maxOutputTokens] - Optional: Max tokens for the response.
 * @returns {Promise<object>} The full stream result object from the Vercel AI SDK.
 * @throws {Error} If the API call fails to initiate the stream.
 */
export async function streamPerplexityText({
	apiKey,
	modelId,
	messages,
	temperature,
	baseUrl,
	contextWindowTokens,
	maxOutputTokens
}) {
	log('debug', `Streaming Perplexity text with model: ${modelId}`);
	try {
		const client = getClient(apiKey, baseUrl);
		const stream = await streamText({
			model: client(modelId),
			messages: messages,
			maxTokens: maxOutputTokens,
			temperature: temperature
		});
		return stream;
	} catch (error) {
		log('error', `Perplexity streamText failed: ${error.message}`);
		throw error;
	}
}

/**
 * Generates a structured object using a Perplexity model.
 * Note: Perplexity API might not directly support structured object generation
 * in the same way as OpenAI or Anthropic. This function might need
 * adjustments or might not be feasible depending on the model's capabilities
 * and the Vercel AI SDK's support for Perplexity in this context.
 *
 * @param {object} params - Parameters for object generation.
 * @param {string} params.apiKey - The Perplexity API key.
 * @param {string} params.modelId - The specific Perplexity model ID.
 * @param {Array<object>} params.messages - The messages array.
 * @param {import('zod').ZodSchema} params.schema - The Zod schema for the object.
 * @param {string} params.objectName - A name for the object/tool.
 * @param {number} [params.temperature] - Temperature for generation.
 * @param {number} [params.maxRetries] - Max retries for validation/generation.
 * @param {string} [params.baseUrl] - Base URL for the Perplexity API.
 * @param {number} [params.contextWindowTokens] - Optional: Max context tokens for the model.
 * @param {number} [params.maxOutputTokens] - Optional: Max tokens for the response.
 * @returns {Promise<object>} The generated object matching the schema and usage.
 * @throws {Error} If generation or validation fails or is unsupported.
 */
export async function generatePerplexityObject({
	apiKey,
	modelId,
	messages,
	schema,
	objectName = 'generated_object',
	temperature,
	maxRetries = 1,
	baseUrl,
	contextWindowTokens,
	maxOutputTokens
}) {
	log(
		'debug',
		`Attempting to generate Perplexity object ('${objectName}') with model: ${modelId}`
	);
	log(
		'warn',
		'generateObject support for Perplexity might be limited or experimental.'
	);
	try {
		const client = getClient(apiKey, baseUrl);
		const result = await generateObject({
			model: client(modelId),
			schema: schema,
			messages: messages,
			maxTokens: maxOutputTokens,
			temperature: temperature,
			maxRetries: maxRetries
		});
		log(
			'debug',
			`Perplexity generateObject result received. Tokens: ${result.usage.completionTokens}/${result.usage.promptTokens}`
		);
		return {
			object: result.object,
			usage: {
				inputTokens: result.usage.promptTokens,
				outputTokens: result.usage.completionTokens
			}
		};
	} catch (error) {
		log(
			'error',
			`Perplexity generateObject ('${objectName}') failed: ${error.message}`
		);
		throw new Error(
			`Failed to generate object with Perplexity: ${error.message}. Structured output might not be fully supported.`
		);
	}
}

// TODO: Implement streamPerplexityObject if needed and feasible.