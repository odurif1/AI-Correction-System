"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth-context";
import {
  Sun,
  Moon,
  Menu,
  X,
  LogOut,
} from "lucide-react";
import { useState } from "react";

export function Header() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const { user, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Navigation items based on auth status
  const publicNavItems: Array<{ href: string; label: string; highlight?: boolean }> = [
    { href: "/", label: "Accueil" },
    { href: "/notre-approche", label: "Notre approche" },
    { href: "/pricing", label: "Tarifs" },
  ];

  const authNavItems: Array<{ href: string; label: string; highlight?: boolean }> = [
    { href: "/dashboard", label: "Mes corrections" },
    { href: "/sessions/new", label: "Nouvelle correction", highlight: true },
    { href: "/subscription", label: "Abonnement" },
  ];

  const navItems = user ? authNavItems : publicNavItems;

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-12 items-center">
        {/* Logo - Left */}
        <div className="flex">
          <Link href={user ? "/sessions/new" : "/"} className="flex items-center gap-2 group">
            <svg
              viewBox="0 0 32 32"
              className="w-6 h-6"
              xmlns="http://www.w3.org/2000/svg"
            >
              <circle cx="16" cy="16" r="15" className="fill-purple-600 dark:fill-purple-500"/>
              <path
                d="M8 16l5 5 11-11"
                className="stroke-white"
                strokeWidth="3.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </svg>
            <span className="font-logo text-[1.5rem] font-medium tracking-normal text-purple-600 dark:text-purple-400 group-hover:text-purple-700 dark:group-hover:text-purple-300 transition-colors duration-200">
              La Corrigeuse
            </span>
          </Link>
        </div>

        {/* Navigation - Center */}
        <nav className="hidden md:flex flex-1 items-center justify-center space-x-6 text-sm font-medium">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "transition-colors",
                item.highlight
                  ? "text-purple-600 dark:text-purple-400 font-semibold hover:text-purple-700 dark:hover:text-purple-300"
                  : cn(
                      "hover:text-foreground/80",
                      pathname === item.href
                        ? "text-foreground"
                        : "text-foreground/60"
                    )
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Auth buttons - Right */}
        <div className="flex items-center space-x-2">
          <div className="hidden md:flex items-center space-x-2">
            {user ? (
              <>
                <span className="text-sm text-muted-foreground mr-2">
                  {user.email}
                </span>
                <Button variant="ghost" size="sm" onClick={logout}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Déconnexion
                </Button>
              </>
            ) : (
              <>
                <Button variant="ghost" size="sm" asChild>
                  <Link href="/auth/login">Connexion</Link>
                </Button>
                <Button size="sm" asChild>
                  <Link href="/auth/register">Inscription</Link>
                </Button>
              </>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Changer le thème</span>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>
      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t">
          <nav className="flex flex-col p-4 space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "px-4 py-2 rounded-md transition-colors",
                  item.highlight
                    ? "text-purple-600 dark:text-purple-400 font-semibold bg-purple-50 dark:bg-purple-950/30"
                    : cn(
                        "hover:bg-muted",
                        pathname === item.href ? "bg-muted" : ""
                      )
                )}
                onClick={() => setMobileMenuOpen(false)}
              >
                {item.label}
              </Link>
            ))}
            <div className="border-t pt-4 mt-2 flex gap-2">
              {user ? (
                <Button variant="ghost" className="flex-1" onClick={() => { logout(); setMobileMenuOpen(false); }}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Déconnexion
                </Button>
              ) : (
                <>
                  <Button variant="ghost" className="flex-1" asChild>
                    <Link href="/auth/login">Connexion</Link>
                  </Button>
                  <Button className="flex-1" asChild>
                    <Link href="/auth/register">Inscription</Link>
                  </Button>
                </>
              )}
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}
