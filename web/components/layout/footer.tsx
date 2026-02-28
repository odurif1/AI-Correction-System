import Link from "next/link";
import { Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t py-4 md:py-0">
      <div className="container flex flex-col items-center justify-between gap-4 md:h-12 md:flex-row">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>&copy; {new Date().getFullYear()} La Corrigeuse</span>
          <span className="hidden sm:inline">•</span>
          <span className="hidden sm:flex items-center gap-1">
            Fait avec <Heart className="h-3 w-3 text-red-500 fill-red-500" /> pour les professeurs
          </span>
        </div>
        <div className="flex items-center gap-6">
          <Link
            href="mailto:contact@lacorrigeuse.fr"
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            Contact
          </Link>
          <Link
            href="#"
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            Mentions légales
          </Link>
        </div>
      </div>
    </footer>
  );
}
