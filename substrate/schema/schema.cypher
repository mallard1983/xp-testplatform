// Experience Engine — Neo4j Schema
// Run via deploy_schema.py. All statements use IF NOT EXISTS — safe to run multiple times.

// ─── Unique Constraints on Primary Keys ──────────────────────────────────────
// NOTE: The implementation plan specifies IS NODE KEY for these, but NODE KEY
// requires Neo4j Enterprise Edition. We are running neo4j:5-community.
// IS UNIQUE is used instead — uniqueness is still enforced. Not-null enforcement
// is handled at the application layer (Pydantic validation on all write paths).
// Decision recorded in docs/ARCHITECTURE.md.

CREATE CONSTRAINT thought_stream_id IF NOT EXISTS
FOR (n:ThoughtStream) REQUIRE (n.id) IS UNIQUE;

CREATE CONSTRAINT topic_id IF NOT EXISTS
FOR (n:Topic) REQUIRE (n.id) IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (n:Entity) REQUIRE (n.id) IS UNIQUE;

// ─── Unique Constraints ───────────────────────────────────────────────────────

CREATE CONSTRAINT topic_name_unique IF NOT EXISTS
FOR (n:Topic) REQUIRE (n.name) IS UNIQUE;

CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS
FOR (n:Entity) REQUIRE (n.name, n.entity_type) IS UNIQUE;

// ─── Vector Index ─────────────────────────────────────────────────────────────

CREATE VECTOR INDEX thought_stream_embedding IF NOT EXISTS
FOR (n:ThoughtStream) ON (n.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
}};

// ─── Fulltext Index ───────────────────────────────────────────────────────────

CREATE FULLTEXT INDEX thought_stream_content IF NOT EXISTS
FOR (n:ThoughtStream) ON EACH [n.content];
