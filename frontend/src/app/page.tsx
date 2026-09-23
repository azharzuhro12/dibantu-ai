import { redirect } from "next/navigation";

/** The dashboard opens on the assistant chat. */
export default function Home() {
  redirect("/chat");
}
