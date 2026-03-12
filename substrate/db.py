"""
All Neo4j database operations for the substrate API.
Route handlers call these functions — no Cypher lives in api.py.
"""

import uuid
from datetime import datetime, timezone

import neo4j
from neo4j import AsyncDriver
from neo4j.exceptions import ConstraintError


# ── ThoughtStream ──────────────────────────────────────────────────────────────

async def create_thought_stream(driver: AsyncDriver, data: dict) -> dict:
    """
    Insert a ThoughtStream node with a pre-generated embedding.
    Returns the created node as a dict. Raises ConstraintError on duplicate id.
    """
    node_id = data.get("id") or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    records, _, _ = await driver.execute_query(
        """
        CREATE (t:ThoughtStream {
            id:               $id,
            content:          $content,
            summary:          $summary,
            created_at:       $created_at,
            source:           $source,
            source_id:        $source_id,
            completion_state: $completion_state,
            confidence:       $confidence
        })
        WITH t
        CALL db.create.setNodeVectorProperty(t, 'embedding', $embedding)
        RETURN t.id AS id
        """,
        id=node_id,
        content=data["content"],
        summary=data["summary"],
        created_at=now,
        source=data["source"],
        source_id=data.get("source_id", ""),
        completion_state=data["completion_state"],
        confidence=float(data["confidence"]),
        embedding=data["embedding"],
        routing_=neo4j.RoutingControl.WRITE,
        database_="neo4j",
    )

    created_id = records[0]["id"]

    if data.get("topic"):
        await _link_to_topic(driver, created_id, data["topic"])

    return await get_thought_stream(driver, created_id)


async def _link_to_topic(driver: AsyncDriver, stream_id: str, topic_name: str) -> None:
    """Create BELONGS_TO relationship if the topic exists."""
    await driver.execute_query(
        """
        MATCH (t:ThoughtStream {id: $stream_id})
        MATCH (tp:Topic {name: $topic_name})
        MERGE (t)-[:BELONGS_TO]->(tp)
        """,
        stream_id=stream_id,
        topic_name=topic_name,
        routing_=neo4j.RoutingControl.WRITE,
        database_="neo4j",
    )


async def get_thought_stream(driver: AsyncDriver, stream_id: str) -> dict | None:
    """Retrieve a ThoughtStream by id, including typed related streams. Returns None if not found."""
    records, _, _ = await driver.execute_query(
        """
        MATCH (t:ThoughtStream {id: $id})
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(tp:Topic)
        WITH t, collect(tp.name) AS topics
        OPTIONAL MATCH (t)-[rel:RELATES_TO|REFINES|CONTRADICTS|PRECEDES]->(related:ThoughtStream)
        RETURN t.id               AS id,
               t.content          AS content,
               t.summary          AS summary,
               t.created_at       AS created_at,
               t.source           AS source,
               t.source_id        AS source_id,
               t.completion_state AS completion_state,
               t.confidence       AS confidence,
               topics,
               collect(DISTINCT CASE WHEN related IS NOT NULL THEN {
                   id:                related.id,
                   summary:           related.summary,
                   confidence:        related.confidence,
                   topics:            [(related)-[:BELONGS_TO]->(rtp:Topic) | rtp.name],
                   relationship_type: type(rel)
               } END) AS related_streams
        """,
        id=stream_id,
        routing_=neo4j.RoutingControl.READ,
        database_="neo4j",
    )
    if not records:
        return None
    record = records[0]
    result = _stream_record_to_dict(record)
    result["related_streams"] = [
        {
            "id":                r["id"],
            "summary":           r["summary"],
            "confidence":        float(r["confidence"]) if r["confidence"] is not None else 0.0,
            "topics":            [t for t in (r["topics"] or []) if t],
            "relationship_type": r["relationship_type"],
        }
        for r in (record["related_streams"] or [])
        if r is not None
    ]
    return result


async def search_streams(
    driver: AsyncDriver,
    embedding: list[float],
    limit: int = 5,
    topic: str | None = None,
) -> list[dict]:
    """Semantic vector search. Optionally filtered by topic name."""
    if topic:
        records, _, _ = await driver.execute_query(
            """
            CALL db.index.vector.queryNodes('thought_stream_embedding', $limit, $embedding)
            YIELD node AS t, score
            MATCH (t)-[:BELONGS_TO]->(tp:Topic {name: $topic})
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(alltp:Topic)
            RETURN t.id               AS id,
                   t.content          AS content,
                   t.summary          AS summary,
                   t.created_at       AS created_at,
                   t.source           AS source,
                   t.source_id        AS source_id,
                   t.completion_state AS completion_state,
                   t.confidence       AS confidence,
                   collect(alltp.name) AS topics,
                   score
            ORDER BY score DESC
            """,
            limit=limit,
            embedding=embedding,
            topic=topic,
            routing_=neo4j.RoutingControl.READ,
            database_="neo4j",
        )
    else:
        records, _, _ = await driver.execute_query(
            """
            CALL db.index.vector.queryNodes('thought_stream_embedding', $limit, $embedding)
            YIELD node AS t, score
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(tp:Topic)
            RETURN t.id               AS id,
                   t.content          AS content,
                   t.summary          AS summary,
                   t.created_at       AS created_at,
                   t.source           AS source,
                   t.source_id        AS source_id,
                   t.completion_state AS completion_state,
                   t.confidence       AS confidence,
                   collect(tp.name)   AS topics,
                   score
            ORDER BY score DESC
            """,
            limit=limit,
            embedding=embedding,
            routing_=neo4j.RoutingControl.READ,
            database_="neo4j",
        )

    return [_search_record_to_dict(r) for r in records]


async def get_recent_streams(
    driver: AsyncDriver,
    n: int = 10,
    source: str | None = None,
) -> list[dict]:
    """Return the most recent N ThoughtStreams, optionally filtered by source."""
    if source:
        records, _, _ = await driver.execute_query(
            """
            MATCH (t:ThoughtStream {source: $source})
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(tp:Topic)
            RETURN t.id               AS id,
                   t.content          AS content,
                   t.summary          AS summary,
                   t.created_at       AS created_at,
                   t.source           AS source,
                   t.source_id        AS source_id,
                   t.completion_state AS completion_state,
                   t.confidence       AS confidence,
                   collect(tp.name)   AS topics
            ORDER BY t.created_at DESC
            LIMIT $n
            """,
            source=source,
            n=n,
            routing_=neo4j.RoutingControl.READ,
            database_="neo4j",
        )
    else:
        records, _, _ = await driver.execute_query(
            """
            MATCH (t:ThoughtStream)
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(tp:Topic)
            RETURN t.id               AS id,
                   t.content          AS content,
                   t.summary          AS summary,
                   t.created_at       AS created_at,
                   t.source           AS source,
                   t.source_id        AS source_id,
                   t.completion_state AS completion_state,
                   t.confidence       AS confidence,
                   collect(tp.name)   AS topics
            ORDER BY t.created_at DESC
            LIMIT $n
            """,
            n=n,
            routing_=neo4j.RoutingControl.READ,
            database_="neo4j",
        )

    return [_stream_record_to_dict(r) for r in records]


# ── Topics ─────────────────────────────────────────────────────────────────────

async def create_topic(driver: AsyncDriver, name: str, description: str = "") -> dict:
    """Create a Topic node. Raises ConstraintError if name already exists."""
    topic_id = str(uuid.uuid4())
    records, _, _ = await driver.execute_query(
        """
        CREATE (tp:Topic {id: $id, name: $name, description: $description})
        RETURN tp.id AS id, tp.name AS name, tp.description AS description
        """,
        id=topic_id,
        name=name,
        description=description,
        routing_=neo4j.RoutingControl.WRITE,
        database_="neo4j",
    )
    return dict(records[0])


async def get_topics(driver: AsyncDriver) -> list[dict]:
    """Return all topics."""
    records, _, _ = await driver.execute_query(
        "MATCH (tp:Topic) RETURN tp.id AS id, tp.name AS name, tp.description AS description",
        routing_=neo4j.RoutingControl.READ,
        database_="neo4j",
    )
    return [dict(r) for r in records]


# ── Admin ──────────────────────────────────────────────────────────────────────

async def reset_database(driver: AsyncDriver) -> int:
    """
    Delete all nodes and relationships. Schema (constraints/indexes) is preserved.
    Returns the number of nodes deleted.
    Called by the orchestrator's neo4j_manager between test runs.
    """
    records, _, _ = await driver.execute_query(
        """
        MATCH (n)
        WITH n, count(n) AS total
        DETACH DELETE n
        RETURN total
        """,
        routing_=neo4j.RoutingControl.WRITE,
        database_="neo4j",
    )
    return records[0]["total"] if records else 0


# ── Health ─────────────────────────────────────────────────────────────────────

async def check_health(driver: AsyncDriver) -> bool:
    """Return True if Neo4j is reachable and responsive."""
    try:
        await driver.verify_connectivity()
        return True
    except Exception:
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _search_record_to_dict(record) -> dict:
    d = _stream_record_to_dict(record)
    d["relevance_score"] = float(record.get("score") or 0.0)
    return d


def _stream_record_to_dict(record) -> dict:
    return {
        "id":               record["id"],
        "content":          record["content"],
        "summary":          record["summary"],
        "created_at":       record["created_at"],
        "source":           record["source"],
        "source_id":        record["source_id"],
        "completion_state": record["completion_state"],
        "confidence":       record["confidence"],
        "topics":           [t for t in record["topics"] if t],
    }
