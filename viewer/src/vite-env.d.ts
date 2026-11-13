/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API base URL for static deploys; unset means same-origin /api. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
