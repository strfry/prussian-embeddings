# Pipeline für Embedding-Stores.  Alle Pfade sind Defaults der CLIs:
#
#   Dictionary: ../corpus/parsed/twanksta_entries.json (kanonische Quelle)
#   Links:      ../fst/build/links.json                (make -C ../fst links)
#   Chunks:     data/embedding_chunks.jsonl
#   Store:      data/$EMBEDDINGS_NAME.*                (= Engine-Store, via mcp/env.<provider>.sh)
#
# Ein-Befehl-Kette (nach `source ../mcp/env.voyage.sh`):
#   make pipeline        # corpus parse → fst links → chunks → store
#
# Danach: MCP-Server neu starten — die Engine lädt Store + Dictionary nur
# beim Start.

.PHONY: pipeline chunks store

pipeline:
	$(MAKE) -C ../corpus twanksta-parse
	$(MAKE) -C ../fst links
	$(MAKE) chunks store

chunks:
	uv run prussian-embeddings-build-chunks --out data/embedding_chunks.jsonl

store:
	uv run prussian-embeddings-generate --chunks data/embedding_chunks.jsonl
