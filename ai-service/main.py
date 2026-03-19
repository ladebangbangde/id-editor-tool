from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from api.detect_api import router as detect_router
from api.generate_api import router as generate_router
from api.health_api import router as health_router
from api.print_api import router as print_router
from api.upload_debug_api import router as upload_debug_router
from core.config import get_settings
from core.exceptions import (
    AppException,
    ERROR_FILE_NOT_FOUND,
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_IMAGE,
    ERROR_PROCESS_FAILED,
)
from utils.file_utils import ensure_upload_dirs
from utils.logger import get_logger, init_logger
from utils.response_utils import error_response

settings = get_settings()
ensure_upload_dirs()
init_logger()
logger = get_logger()

app = FastAPI(title='AI ID Photo Service', version='1.1.0', docs_url=None, redoc_url=None)
app.mount(settings.static_mount_path, StaticFiles(directory=str(settings.upload_root_path), check_dir=False), name='uploads')
app.include_router(health_router)
app.include_router(detect_router)
app.include_router(generate_router)
app.include_router(print_router)
app.include_router(upload_debug_router)


@app.on_event('startup')
def startup_event() -> None:
    ensure_upload_dirs()
    logger.info('AI service startup complete')


@app.exception_handler(AppException)
async def handle_app_exception(_request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc.message, data=exc.data, error_code=exc.error_code),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(_request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else None
    message = first_error.get('msg', 'Invalid request arguments') if first_error else 'Invalid request arguments'
    return JSONResponse(
        status_code=422,
        content=error_response(message, data=None, error_code=ERROR_INVALID_ARGUMENT),
    )


@app.exception_handler(FileNotFoundError)
async def handle_file_not_found(_request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content=error_response(str(exc), data=None, error_code=ERROR_FILE_NOT_FOUND),
    )


@app.exception_handler(UnidentifiedImageError)
async def handle_invalid_image(_request: Request, exc: UnidentifiedImageError):
    return JSONResponse(
        status_code=400,
        content=error_response(str(exc) or 'Invalid image', data=None, error_code=ERROR_INVALID_IMAGE),
    )


@app.exception_handler(ValueError)
async def handle_value_error(_request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=error_response(str(exc), data=None, error_code=ERROR_INVALID_ARGUMENT),
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(_request: Request, exc: Exception):
    logger.exception('Unhandled exception: {}', exc)
    return JSONResponse(
        status_code=500,
        content=error_response('Image processing failed', data=None, error_code=ERROR_PROCESS_FAILED),
    )


@app.get('/')
def root():
    return {'service': settings.app_name, 'status': 'running'}


def _offline_docs_html(page_title: str, heading: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{page_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Arial, Helvetica, sans-serif;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: #0b1020;
      color: #e5e7eb;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    h1, h2, h3 {{
      margin-bottom: 12px;
    }}
    .hint {{
      color: #cbd5e1;
      margin-bottom: 16px;
    }}
    .card {{
      background: #111827;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }}
    .path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      word-break: break-all;
    }}
    .method {{
      display: inline-block;
      min-width: 72px;
      text-align: center;
      border-radius: 999px;
      padding: 4px 10px;
      margin-right: 8px;
      font-weight: 700;
      color: white;
    }}
    .get {{ background: #0284c7; }}
    .post {{ background: #16a34a; }}
    .put {{ background: #ca8a04; }}
    .delete {{ background: #dc2626; }}
    .patch {{ background: #7c3aed; }}
    a {{
      color: #93c5fd;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #020617;
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{heading}</h1>
    <p class="hint">
      This page is served fully from the ai-service process and does not depend on external CDN assets.
      Open the raw schema at <a href="/openapi.json">/openapi.json</a>.
    </p>
    <div id="content" class="card">Loading OpenAPI schema...</div>
  </main>
  <script>
    const content = document.getElementById('content');

    function methodClass(method) {{
      const normalized = method.toLowerCase();
      return ['get', 'post', 'put', 'delete', 'patch'].includes(normalized) ? normalized : '';
    }}

    function renderSchema(schema) {{
      const sections = [];
      const paths = schema.paths || {{}};
      const tagDescriptions = new Map((schema.tags || []).map((tag) => [tag.name, tag.description || '']));
      const grouped = new Map();

      for (const [path, operations] of Object.entries(paths)) {{
        for (const [method, operation] of Object.entries(operations)) {{
          const tags = operation.tags && operation.tags.length ? operation.tags : ['default'];
          for (const tag of tags) {{
            if (!grouped.has(tag)) grouped.set(tag, []);
            grouped.get(tag).push({{ path, method: method.toUpperCase(), operation }});
          }}
        }}
      }}

      if (!grouped.size) {{
        content.innerHTML = '<p>No API paths were found in the schema.</p>';
        return;
      }}

      for (const [tag, items] of grouped.entries()) {{
        sections.push(`<section class="card"><h2>${{tag}}</h2>${{tagDescriptions.get(tag) ? `<p>${{tagDescriptions.get(tag)}}</p>` : ''}}`);
        for (const item of items) {{
          const summary = item.operation.summary || item.operation.description || 'No summary provided.';
          const requestBody = item.operation.requestBody ? `<details><summary>Request body</summary><pre>${{JSON.stringify(item.operation.requestBody, null, 2)}}</pre></details>` : '';
          const responses = item.operation.responses ? `<details><summary>Responses</summary><pre>${{JSON.stringify(item.operation.responses, null, 2)}}</pre></details>` : '';
          sections.push(`
            <article style="margin-top: 14px;">
              <div><span class="method ${{methodClass(item.method)}}">${{item.method}}</span><span class="path">${{item.path}}</span></div>
              <p>${{summary}}</p>
              ${{requestBody}}
              ${{responses}}
            </article>
          `);
        }}
        sections.push('</section>');
      }}

      content.outerHTML = sections.join('');
    }}

    fetch('/openapi.json')
      .then((response) => {{
        if (!response.ok) {{
          throw new Error(`Failed to load schema: ${{response.status}}`);
        }}
        return response.json();
      }})
      .then(renderSchema)
      .catch((error) => {{
        content.innerHTML = `<h2>Failed to load schema</h2><pre>${{error.message}}</pre>`;
      }});
  </script>
</body>
</html>"""


@app.get('/docs', include_in_schema=False)
def offline_docs():
    return HTMLResponse(_offline_docs_html('AI ID Photo Service - Docs', 'AI ID Photo Service Docs'))


@app.get('/redoc', include_in_schema=False)
def offline_redoc():
    return HTMLResponse(_offline_docs_html('AI ID Photo Service - ReDoc', 'AI ID Photo Service API Reference'))
