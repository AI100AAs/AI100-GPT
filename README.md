# AI100-GPT playground

This repository contains the standalone AI100-GPT teaching playground: a very
small GPT model that runs in the browser and makes next-character prediction,
style, and fine-tuning visible through short experiments.

**Try it online:** [AI100-GPT playground](https://ai100-gpt-playground.ai100aas.chatgpt.site)

The playground is designed for learning, not for answering questions like a
chatbot. It continues a starting phrase one character at a time, so students
can see how a model's learned text patterns shape each next choice.

## What is included

- **Explore:** choose a bundled text style and generate a continuation.
- **Teach it your style:** add several lines of your own writing and train a
  small adapter on-device.
- **Compare:** generate the same prompt before and after teaching, then
  continue both outputs with one button.
- **Chat:** see how a very small model behaves after being shown the shape of
  short question-and-answer exchanges, without presenting it as a factual assistant.
- **How big is this?:** compare the classroom model's scale with familiar
  language models.
- **Save and load:** download a trained style as JSON and load it again later.
- **Input context:** provide up to 128 characters as the starting context; the
  default generated sequence length is 200 characters.
- **Dark and light themes:** the selected theme is remembered locally.

The public interface currently exposes the **GPT Micro** checkpoint. Other
small GPT configurations and checkpoints remain in the source for experiments,
but are intentionally hidden from the student flow for now.

## Privacy and hardware

All text processing, model loading, training, and generation happen locally in
the browser. Student writing and generated text are not sent to a server. The
playground uses TensorFlow.js and prefers WebGPU, with WebGL or CPU fallbacks
when needed. No account, API key, backend, hosted model, or installation is
required to use the live playground.

## Responsible use

This is an experimental teaching tool, not a source of advice or reliable
facts. The model can produce unexpected, biased, offensive, or inaccurate text.
Review anything it generates before sharing or acting on it. The project team
is not responsible for generated content or how it is interpreted or used.

## Run locally

Development requires Node.js 20 or newer.

```sh
npm install
npm run playground-web
```

Open the local URL printed by the development server. The source playground
normally uses the `/AI100-GPT` path, so the default address is:

<http://localhost:3000/AI100-GPT>

The repository also contains the original TypeScript model package and a small
Node.js example under `gpt/` and `playground-node/`.

## Build

Build all workspaces with:

```sh
npm run build
```

To build just the web playground:

```sh
npm run build -w @playground/web
```

The compiled web app is written to `playground-web/build/` and can be served by
any static host.

## Development checks

Before publishing changes, run:

```sh
npm run build
npm run build -w @playground/web
```

## Project structure

- `playground-web/src/components/` — the React playground interface.
- `playground-web/public/` — bundled datasets, model weights, and browser assets.
- `gpt/src/model.ts` — the TensorFlow.js GPT implementation.
- `gpt/src/config.ts` — the available model-size configurations.
- `playground-node/` — a Node.js training and generation example.

The project is based on [homemade-gpt-js](https://github.com/trekhleb/homemade-gpt-js)
and has been adapted for the AI100 classroom activity.

## License and acknowledgements

The GPT implementation follows the upstream project's educational code and
references Andrej Karpathy's lecture, [Let's build GPT: from scratch, in code,
spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY). The bundled
Shakespeare text is public domain.
