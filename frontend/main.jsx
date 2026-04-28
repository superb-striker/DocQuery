import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import DocQuery from "./DocQuery";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <DocQuery />
  </StrictMode>
);