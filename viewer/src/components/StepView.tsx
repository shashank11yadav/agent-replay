import type { Span } from "../api/schemas";

type Rec = Record<string, unknown>;

const isRec = (v: unknown): v is Rec => typeof v === "object" && v !== null && !Array.isArray(v);
const asRec = (v: unknown): Rec => (isRec(v) ? v : {});
const asList = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

/** Untrusted payload values are rendered as text only. */
function Json({ value }: { value: unknown }) {
  return <pre>{JSON.stringify(value ?? null, null, 2)}</pre>;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function ModelStep({ span }: { span: Span }) {
  const request = asRec(span.payload.request);
  const response = asRec(span.payload.response);
  const messages = asList(request.messages);
  const toolCalls = asList(response.tool_calls);
  const text = typeof response.text === "string" ? response.text : null;
  return (
    <dl>
      <Field label="Request messages">
        {messages.length === 0 ? (
          "none"
        ) : (
          <ol>
            {messages.map((m, i) => {
              const msg = asRec(m);
              return (
                <li key={i}>
                  <strong>{String(msg.role ?? "")}</strong>
                  {typeof msg.text === "string" && <p>{msg.text}</p>}
                  {asList(msg.tool_calls).length > 0 && <Json value={msg.tool_calls} />}
                  {asList(msg.tool_results).length > 0 && <Json value={msg.tool_results} />}
                </li>
              );
            })}
          </ol>
        )}
      </Field>
      {text !== null && (
        <Field label="Response text">
          <p>{text}</p>
        </Field>
      )}
      {toolCalls.length > 0 && (
        <Field label="Response tool calls">
          <Json value={toolCalls} />
        </Field>
      )}
      <Field label="Input tokens">{span.input_tokens}</Field>
      <Field label="Output tokens">{span.output_tokens}</Field>
      <Field label="Latency (latency_ms)">{span.latency_ms}</Field>
      <Field label="Cache status">{span.cache_status ?? "none"}</Field>
    </dl>
  );
}

function ToolStep({ span }: { span: Span }) {
  const fault = span.payload.fault;
  return (
    <dl>
      <Field label="Arguments">
        <Json value={span.payload.args} />
      </Field>
      <Field label="Result">
        <Json value={span.payload.result} />
      </Field>
      <Field label="Latency (latency_ms)">{span.latency_ms}</Field>
      {fault !== null && fault !== undefined && <Field label="Fault">{String(fault)}</Field>}
    </dl>
  );
}

export function StepView({ span }: { span: Span }) {
  return (
    <section aria-label="Current step">
      <h2>{span.name}</h2>
      <p>Kind: {span.kind}</p>
      {span.status !== "ok" && <p>Status: {span.status}</p>}
      {span.error_type && <p>Error type: {span.error_type}</p>}
      {span.kind === "model" ? (
        <ModelStep span={span} />
      ) : span.kind === "tool" ? (
        <ToolStep span={span} />
      ) : (
        <Json value={span.payload} />
      )}
    </section>
  );
}
