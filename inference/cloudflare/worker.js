import { Container } from '@cloudflare/containers';
import { routeRequest, serveContainer } from './routing.js';

export class PriestContainer extends Container {
  defaultPort = 8080;
  sleepAfter = '30s';
  enableInternet = false;
  pingEndpoint = 'localhost/health';
  envVars = { MODEL_DIR: '/app/model', CPU_THREADS: '1', REQUEST_CAPACITY: '3' };

  async fetch(request) {
    return serveContainer(this, request, this.env.MODEL_SHA256);
  }
}

export default { fetch: routeRequest };
