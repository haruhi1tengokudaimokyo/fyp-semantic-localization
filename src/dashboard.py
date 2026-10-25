"""Read-only localhost report server; no external uploads or exposure."""
import json,csv,io,argparse,os
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
BASE=Path(__file__).resolve().parents[1]
RUN=Path(os.environ.get('FYP_ROOT','/data/fyp-semrayloc'))/'runs/smoke'
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw): super().__init__(*a,directory=str(BASE),**kw)
    def do_GET(self):
        if self.path.split('?')[0]=='/api/status':
            state=json.loads((RUN/'status.json').read_text()) if (RUN/'status.json').exists() else {'state':'waiting_for_dependencies_or_weights','validation_available':False}
            rows=list(csv.DictReader(io.StringIO((RUN/'metrics.csv').read_text()))) if (RUN/'metrics.csv').exists() else []
            payload=json.dumps({'status':state,'rows':rows},ensure_ascii=False).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(payload)
        elif self.path=='/': self.path='/dashboard.html'; super().do_GET()
        else: super().do_GET()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args();ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
