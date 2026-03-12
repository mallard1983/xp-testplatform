#!/usr/bin/env python3
"""
Deploy the Neo4j schema (constraints + indexes) against a running Neo4j instance.
Idempotent — safe to run multiple times. All DDL uses IF NOT EXISTS.
"""

import os
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


SCHEMA_FILE = Path(__file__).parent / "schema.cypher"


def get_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth_raw = os.environ.get("NEO4J_AUTH", "neo4j/changeme")
    user, password = auth_raw.split("/", 1)
    return GraphDatabase.driver(uri, auth=(user, password))


def parse_statements(cypher: str) -> list[str]:
    """Split cypher file into individual statements, stripping comments and blanks."""
    statements = []
    current = []
    for line in cypher.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or not stripped:
            continue
        current.append(stripped)
        if stripped.endswith(";"):
            stmt = " ".join(current).rstrip(";").strip()
            if stmt:
                statements.append(stmt)
            current = []
    return statements


def deploy(driver) -> list[str]:
    cypher = SCHEMA_FILE.read_text()
    statements = parse_statements(cypher)
    deployed = []

    with driver.session(database="neo4j") as session:
        for stmt in statements:
            session.run(stmt)
            # Extract the identifier from the statement for reporting
            label = stmt.split()[2] if len(stmt.split()) > 2 else stmt[:60]
            deployed.append(label)
            print(f"  OK  {label}")

    return deployed


def wait_for_neo4j(driver, retries=10, delay=3):
    for attempt in range(1, retries + 1):
        try:
            driver.verify_connectivity()
            return
        except ServiceUnavailable:
            print(f"  Neo4j not ready (attempt {attempt}/{retries}), waiting {delay}s...")
            time.sleep(delay)
    print("ERROR: Neo4j did not become available in time.", file=sys.stderr)
    sys.exit(1)


def main():
    print("=== deploy_schema.py ===")
    driver = get_driver()
    try:
        print("Waiting for Neo4j...")
        wait_for_neo4j(driver)
        print(f"Connected. Deploying schema from {SCHEMA_FILE.name}...")
        deployed = deploy(driver)
        print(f"\nDeployed {len(deployed)} statements. Schema is up to date.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
