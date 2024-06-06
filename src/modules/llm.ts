import { ChatOpenAI } from "@langchain/openai";

// TODO: Implement
export const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  model:'gpt-4o'
});
