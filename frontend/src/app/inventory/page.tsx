import type { Metadata } from "next";

import { InventoryView } from "@/components/inventory-view";

export const metadata: Metadata = {
  title: "Inventory",
};

export default function InventoryPage() {
  return <InventoryView />;
}
