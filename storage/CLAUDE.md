# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Cloud storage upload after optimization. Currently only Google Cloud Storage (GCS).

## Design

`GCSUploader` is a module-level singleton with lazy client initialization. Authentication is automatic on Cloud Run (workload identity) and via `GOOGLE_APPLICATION_CREDENTIALS` locally.

Storage is optional — the `/optimize` endpoint only uploads when a `storage` config is provided in the request. Without it, optimized bytes are returned directly in the response body.

## Adding a New Provider

The schema (`schemas.py`) currently restricts `StorageConfig.provider` to `"gcs"` via regex. To add a provider: create a new uploader class, update the regex, and dispatch in `routers/optimize.py`'s `_build_json_response`.
