import { detectCommand } from "./commands";
import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnablePick, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { llm } from "../llm";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";

// tag::call[]
export async function call(
  message: string,
  sessionId: string
): Promise<string> {

  // embedding model
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    model: 'text-embedding-ada-002' // 1536 dimensions , 12.5 k pages per dollar
  })

  // Create vector store
  const store = await Neo4jVectorStore.fromExistingGraph(embeddings, {
    url: process.env.NEO4J_URI,
    username: process.env.NEO4J_USERNAME,
    password: process.env.NEO4J_PASSWORD,
    nodeLabel: "Talk",
    textNodeProperties: ["title", "description"],
    indexName: "talk_embeddings_openai",
    embeddingNodeProperty: "embedding",
    retrievalQuery: `
      RETURN node.description AS text, score,
        node {
          .time, .title,
          url: 'https://athens.cityjsconf.org/'+ node.url,
          speaker: [
            (node)-[:GIVEN_BY]->(s) |
            s { .name, .company, .x_handle, .bio }
            ][0],
          room: [ (node)-[:IN_ROOM]->(r) | r.name ][0],
          tags: [ (node)-[:HAS_TAG]->(t) | t.name ]

        } AS metadata
`,
  });

  const retriever = store.asRetriever(3)
  // Detect slash commands
  const command = detectCommand(message, sessionId);

  if (typeof command === "string") {
    return command;
  }

  const prompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(`You are a helpful assistant helping users with queries about the CityJS`),
    SystemMessagePromptTemplate.fromTemplate(
      `
      Here are some talks to help you answer the question.
      Don't use your pre-trained knowledge to answer the question.
      Always include a full link to the meetup.
      If the answer isn't included in the documents, say you don't know.

      Documents:
      {documents}
    `
    ),
    HumanMessagePromptTemplate.fromTemplate(`Question: {message}`)
  ])

  const chain = RunnableSequence.from<{message: string}, string>([RunnablePassthrough.assign({ 
    documents: new RunnablePick("message").pipe(retriever.pipe(docs => JSON.stringify(docs)))
  }),prompt, llm, new StringOutputParser])

  const output = await chain.invoke({message})

  return output;
}
// end::call[]
