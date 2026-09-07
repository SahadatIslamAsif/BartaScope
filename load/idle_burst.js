// Profile 2: idle, then a sudden burst.
//
// The intended use is the scale-to-zero story on Cloud Run with
// min-instances 0, where the burst arrives at a service that has no
// instances running and the first requests pay the full cold start.
//
// Run locally against an already-running container it measures something
// weaker but still real: what a burst costs when the process is alive but
// its thread pools and allocator have gone quiet. ONNX Runtime's intra-op
// pool spin-waits briefly and then parks its threads, and the allocator
// arena is cold, so the first requests after an idle period are slower than
// steady state. Do not report a local run of this as a cold-start number --
// nothing here is scaling from zero.
//
// The idle phase is 1 rps rather than literally zero so that the run has a
// baseline to compare the burst against, from the same process, seconds
// apart.

import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

const messages = new SharedArray('messages', () =>
  JSON.parse(open('/work/messages.json')));

const BASE_URL = __ENV.BASE_URL || 'http://bartascope-api:8080';
const IDLE_SECONDS = Number(__ENV.IDLE_SECONDS || 60);
const IDLE_RATE = Number(__ENV.IDLE_RATE || 1);
const BURST_SECONDS = Number(__ENV.BURST_SECONDS || 30);
const BURST_RATE = Number(__ENV.BURST_RATE || 100);

export const options = {
  scenarios: {
    idle: {
      executor: 'constant-arrival-rate',
      rate: IDLE_RATE,
      timeUnit: '1s',
      duration: `${IDLE_SECONDS}s`,
      preAllocatedVUs: 5,
      maxVUs: 20,
      startTime: '0s',
      tags: { phase: 'idle' },
      exec: 'classify',
    },
    burst: {
      executor: 'constant-arrival-rate',
      rate: BURST_RATE,
      timeUnit: '1s',
      duration: `${BURST_SECONDS}s`,
      preAllocatedVUs: Math.max(50, BURST_RATE),
      maxVUs: Math.max(200, BURST_RATE * 4),
      // Starts the instant the idle phase ends. No ramp: the point is the
      // step change.
      startTime: `${IDLE_SECONDS}s`,
      tags: { phase: 'burst' },
      exec: 'classify',
    },
  },
  thresholds: {
    'http_req_duration{phase:idle}':
      [{ threshold: 'p(95)<600000', abortOnFail: false }],
    'http_req_duration{phase:burst}':
      [{ threshold: 'p(95)<600000', abortOnFail: false }],
    'http_req_failed{phase:idle}':
      [{ threshold: 'rate<=1', abortOnFail: false }],
    'http_req_failed{phase:burst}':
      [{ threshold: 'rate<=1', abortOnFail: false }],
    'iterations{phase:idle}':
      [{ threshold: 'count>=0', abortOnFail: false }],
    'iterations{phase:burst}':
      [{ threshold: 'count>=0', abortOnFail: false }],
    'dropped_iterations{phase:burst}':
      [{ threshold: 'count>=0', abortOnFail: false }],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(95)', 'p(99)', 'max', 'count'],
};

export function classify() {
  const text = messages[Math.floor(Math.random() * messages.length)];
  const res = http.post(`${BASE_URL}/v1/classify`,
    JSON.stringify({ text: text }),
    { headers: { 'Content-Type': 'application/json' } });
  check(res, { 'status 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  return { '/work/summary_burst.json': JSON.stringify(data, null, 2) };
}
