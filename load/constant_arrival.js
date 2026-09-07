// Profile 1: constant arrival rate, stepped, to find where p95 breaks.
//
// Constant *arrival rate* rather than constant VUs, deliberately. With a
// fixed VU count, each VU waits for its response before sending the next
// request, so the offered load quietly drops as the service slows down and
// the test can never show you saturation -- it shows you a queue that
// politely forms itself. Arrival rate keeps sending at the target regardless,
// which is what real traffic does.
//
// Each rate is its own scenario with its own tag, so the summary reports
// per-rate percentiles instead of one number blended across the whole ramp.
// The thresholds are set to values that cannot fail on purpose: they exist
// only because k6 computes tagged sub-metrics for thresholds, which is how
// the per-rate percentiles get into the summary at all.
//
// `dropped_iterations` is the metric that matters most here. When it goes
// non-zero the load generator could not launch requests fast enough at the
// target rate, which means the service is already saturated and the reported
// latency for that step understates the problem.

import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';
import { Counter, Trend } from 'k6/metrics';

// A 503 from the load shedder is the service working correctly under
// overload, not a failure. Counting it as one would make a healthy shedding
// service look broken, and would also bury the thing worth measuring: how
// fast the requests it DID serve came back.
http.setResponseCallback(http.expectedStatuses(200, 503));

// Latency of served requests only. Shed responses return in microseconds, so
// blending them into http_req_duration would drag p95 down and make an
// overloaded service look faster than a healthy one.
const served = new Trend('served_duration', true);
const shed = new Counter('shed');

const messages = new SharedArray('messages', () =>
  JSON.parse(open('/work/messages.json')));

const BASE_URL = __ENV.BASE_URL || 'http://bartascope-api:8080';
const RATES = (__ENV.RATES || '20,40,60,80,100,120,140,160')
  .split(',').map(Number);
const STAGE_SECONDS = Number(__ENV.STAGE_SECONDS || 30);
// A gap between steps so a slow tail from one rate cannot be counted against
// the next one.
const GAP_SECONDS = Number(__ENV.GAP_SECONDS || 5);

// Unmeasured warm-up. Without it the first rate step absorbs the cost of a
// process that has just started: ORT's thread pool has not spun up, its
// allocator arena is cold, and Python is still importing on first touch. That
// showed up as 85 rps reporting a worse p95 than 90 and 95 rps, which is not
// a capacity result, it is a cold container. Tagged separately so none of it
// reaches the per-rate metrics.
const WARMUP_SECONDS = Number(__ENV.WARMUP_SECONDS || 15);
const WARMUP_RATE = Number(__ENV.WARMUP_RATE || 20);
const OFFSET = WARMUP_SECONDS + GAP_SECONDS;

const scenarios = {
  warmup: {
    executor: 'constant-arrival-rate',
    rate: WARMUP_RATE,
    timeUnit: '1s',
    duration: `${WARMUP_SECONDS}s`,
    preAllocatedVUs: 20,
    maxVUs: 50,
    startTime: '0s',
    tags: { phase: 'warmup' },
    exec: 'classify',
  },
};
const thresholds = {};

RATES.forEach((rate, i) => {
  scenarios[`rate_${rate}`] = {
    executor: 'constant-arrival-rate',
    rate: rate,
    timeUnit: '1s',
    duration: `${STAGE_SECONDS}s`,
    // Enough VUs that the executor is never the bottleneck. If these run out
    // k6 reports dropped_iterations, which we surface rather than hide.
    preAllocatedVUs: Math.max(20, rate),
    maxVUs: Math.max(100, rate * 4),
    startTime: `${OFFSET + i * (STAGE_SECONDS + GAP_SECONDS)}s`,
    tags: { rate: String(rate) },
    exec: 'classify',
  };
  thresholds[`http_req_duration{rate:${rate}}`] =
    [{ threshold: 'p(95)<600000', abortOnFail: false }];
  thresholds[`http_req_failed{rate:${rate}}`] =
    [{ threshold: 'rate<=1', abortOnFail: false }];
  thresholds[`iterations{rate:${rate}}`] =
    [{ threshold: 'count>=0', abortOnFail: false }];
  thresholds[`dropped_iterations{rate:${rate}}`] =
    [{ threshold: 'count>=0', abortOnFail: false }];
  thresholds[`served_duration{rate:${rate}}`] =
    [{ threshold: 'p(95)<600000', abortOnFail: false }];
  thresholds[`shed{rate:${rate}}`] =
    [{ threshold: 'count>=0', abortOnFail: false }];
});

export const options = {
  scenarios,
  thresholds,
  summaryTrendStats: ['avg', 'min', 'med', 'p(95)', 'p(99)', 'max', 'count'],
  discardResponseBodies: false,
};

export function classify() {
  const text = messages[Math.floor(Math.random() * messages.length)];
  const res = http.post(`${BASE_URL}/v1/classify`,
    JSON.stringify({ text: text }),
    { headers: { 'Content-Type': 'application/json' } });
  if (res.status === 200) {
    served.add(res.timings.duration);
  } else if (res.status === 503) {
    shed.add(1);
  }
  check(res, {
    'served or shed': (r) => r.status === 200 || r.status === 503,
    'has label when served': (r) =>
      r.status !== 200 || r.json('label') !== undefined,
  });
}

export function handleSummary(data) {
  return { '/work/summary_constant.json': JSON.stringify(data, null, 2) };
}
