"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import {
  Heart,
  Clock,
  ShieldCheck,
  ArrowRight,
  CheckCircle2,
  MessageSquare,
} from "lucide-react";
import { useState, useEffect } from "react";

const slogans = [
  "Elle s'occupe de vos copies.",
  "Elle fait comme vous. Juste plus vite.",
  "La rigueur d'un prof. La vitesse de l'IA.",
];

const howItWorks = [
  {
    step: "1",
    title: "Téléversez vos copies",
    description: "Glissez vos PDF, c'est prêt. Importez toute une classe d'un coup.",
  },
  {
    step: "2",
    title: "Laissez faire La Corrigeuse",
    description: "Deux IA corrigent chaque copie et signalent les points à vérifier.",
  },
  {
    step: "3",
    title: "Validez et exportez",
    description: "Consultez, ajustez si besoin, puis exportez vers Pronote ou Excel.",
  },
];

export default function LandingPage() {
  const [slogan, setSlogan] = useState(slogans[0]);

  useEffect(() => {
    setSlogan(slogans[Math.floor(Math.random() * slogans.length)]);
  }, []);

  // Scroll reveal effect
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed");
          }
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll(".scroll-reveal").forEach((el) => {
      observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative overflow-hidden">
          {/* Animated background gradient */}
          <div className="absolute inset-0 bg-gradient-to-br from-purple-50 via-blue-50 to-purple-50 dark:from-purple-950/20 dark:via-blue-950/20 dark:to-purple-950/20 animated-gradient" />

          <div className="relative container py-16 md:py-24">
            <div className="max-w-4xl mx-auto text-center space-y-6">
              {/* Main headline */}
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-balance">
                La Corrigeuse que tout{" "}
                <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  prof
                </span>{" "}
                rêve d'avoir.
              </h1>

              {/* Subheadline */}
              <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
                {slogan}
              </p>

              {/* CTA buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
                <Button size="lg" className="text-lg px-8 py-6 bg-purple-600 hover:bg-purple-700" asChild>
                  <Link href="/auth/register">
                    Essayer gratuitement
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Link>
                </Button>
                <Button size="lg" variant="outline" className="text-lg px-8 py-6" asChild>
                  <Link href="/pricing">Voir les tarifs</Link>
                </Button>
              </div>

              {/* Trust indicators */}
              <p className="text-sm text-muted-foreground pt-4">
                Corrige et annote vos copies
              </p>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="container py-16 md:py-24 scroll-reveal">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Un collègue qui ne vous déçoit jamais
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              La Corrigeuse travaille avec la rigueur que vous apportez à chaque copie.
            </p>
          </div>

          <div className="space-y-16 md:space-y-24">
            {/* Feature 1: Double IA */}
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="space-y-4">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 text-sm font-medium">
                  <ShieldCheck className="h-4 w-4" />
                  Fiable
                </div>
                <h3 className="text-2xl md:text-3xl font-bold">
                  Deux IA, une seule vérité
                </h3>
                <p className="text-muted-foreground text-lg">
                  Deux modèles différents analysent chaque copie indépendamment.
                  S'ils sont d'accord, la note est validée. S'ils divergent,
                  vous êtes notifié pour trancher.
                </p>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>95% de précision sur les notes finales (sujets de référence)</span>
                  </li>
                  <li className="flex items-center gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>Les désaccords sont signalés automatiquement</span>
                  </li>
                </ul>
              </div>
              <div className="relative">
                {/* Mock UI: Double validation */}
                <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-950/30 dark:to-blue-950/30 rounded-2xl p-6 shadow-xl">
                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow-lg space-y-3">
                    <div className="text-sm font-medium text-muted-foreground mb-2">Question 3 - Analyse</div>
                    <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                          <CheckCircle2 className="h-4 w-4 text-white" />
                        </div>
                        <span className="text-sm font-medium">LLM1</span>
                      </div>
                      <span className="font-bold text-green-600">4/5</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
                          <CheckCircle2 className="h-4 w-4 text-white" />
                        </div>
                        <span className="text-sm font-medium">LLM2</span>
                      </div>
                      <span className="font-bold text-green-600">4/5</span>
                    </div>
                    <div className="border-t pt-3 flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Consensus</span>
                      <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-sm font-medium">
                        Validé automatiquement
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Feature 2: Annotations */}
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="order-2 md:order-1 relative">
                {/* Mock UI: Annotation */}
                <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-950/30 dark:to-blue-950/30 rounded-2xl p-6 shadow-xl">
                  <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow-lg">
                    <div className="space-y-3">
                      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
                      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
                      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
                      <div className="relative mt-4">
                        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
                        <div className="absolute -left-2 top-1/2 -translate-y-1/2 w-1 h-5 bg-red-400 rounded"></div>
                        <div className="absolute -right-2 top-0 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 text-xs px-2 py-1 rounded shadow-sm">
                          <MessageSquare className="h-3 w-3 inline mr-1" />
                          Mal formulé
                        </div>
                      </div>
                      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-4/5"></div>
                      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
                    </div>
                    <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <p className="text-sm text-purple-700 dark:text-purple-300">
                        "Bonne analyse, mais la conclusion pourrait être plus développée."
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="order-1 md:order-2 space-y-4">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 text-sm font-medium">
                  <Heart className="h-4 w-4" />
                  Humain
                </div>
                <h3 className="text-2xl md:text-3xl font-bold">
                  Annoté comme un vrai prof
                </h3>
                <p className="text-muted-foreground text-lg">
                  Chaque copie reçoit des commentaires personnalisés,
                  des soulignements et des annotations.
                  Exactement comme vous le feriez avec votre stylo rouge.
                </p>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>Commentaires adaptés au niveau de l'élève</span>
                  </li>
                  <li className="flex items-center gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>Soulignements des passages importants</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Feature 3: Gain de temps */}
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="space-y-4">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 text-sm font-medium">
                  <Clock className="h-4 w-4" />
                  Rapide
                </div>
                <h3 className="text-2xl md:text-3xl font-bold">
                  Récupérez vos week-end
                </h3>
                <p className="text-muted-foreground text-lg">
                  Laissez l'IA préparer la correction, vous validez.
                  Simple et efficace. Ce qui prenait 3 heures ne prend plus que 30 minutes.
                </p>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>200 copies traitées en 1 heure</span>
                  </li>
                  <li className="flex items-center gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>Export vers Pronote, Excel, PDF</span>
                  </li>
                </ul>
              </div>
              <div className="relative">
                {/* Mock UI: Time comparison */}
                <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-950/30 dark:to-blue-950/30 rounded-2xl p-6 shadow-xl">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow-lg text-center">
                      <div className="text-3xl font-bold text-red-500 mb-1">3h</div>
                      <div className="text-xs text-muted-foreground">Sans La Corrigeuse</div>
                      <div className="mt-3 h-2 bg-red-100 dark:bg-red-900/30 rounded-full">
                        <div className="h-full bg-red-400 rounded-full w-full"></div>
                      </div>
                    </div>
                    <div className="bg-white dark:bg-gray-900 rounded-xl p-4 shadow-lg text-center border-2 border-purple-500">
                      <div className="text-3xl font-bold text-purple-600 mb-1">30min</div>
                      <div className="text-xs text-muted-foreground">Avec La Corrigeuse</div>
                      <div className="mt-3 h-2 bg-purple-100 dark:bg-purple-900/30 rounded-full">
                        <div className="h-full bg-purple-500 rounded-full w-1/6"></div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 text-center">
                    <span className="text-2xl font-bold text-green-600">-83%</span>
                    <span className="text-sm text-muted-foreground ml-2">de temps de correction</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How it Works */}
        <section className="bg-muted/30 scroll-reveal">
          <div className="container py-16">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                Simple comme un copier-coller
              </h2>
              <p className="text-lg text-muted-foreground">
                Trois étapes. C'est tout.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
              {howItWorks.map((item, index) => (
                <div key={item.step} className="relative">
                  {index < howItWorks.length - 1 && (
                    <div className="hidden md:block absolute top-10 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-purple-300 to-transparent" />
                  )}
                  <div className="flex flex-col items-center text-center">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 text-white flex items-center justify-center text-2xl font-bold mb-6 shadow-lg">
                      {item.step}
                    </div>
                    <h3 className="font-semibold text-lg mb-2">{item.title}</h3>
                    <p className="text-sm text-muted-foreground max-w-xs">
                      {item.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {/* Link to approach page */}
            <div className="text-center mt-12">
              <Button variant="outline" asChild>
                <Link href="/notre-approche">
                  Comment ça marche ?
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="relative overflow-hidden scroll-reveal">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-600 to-blue-600 animated-gradient" />
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRjMC0yLjIwOS0xLjc5MS00LTQtNHMtNCAxLjc5MS00IDQgMS43OTEgNCA0IDQgNC0xLjc5MSA0LTR6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-30" />

          <div className="relative container py-16">
            <div className="max-w-2xl mx-auto text-center text-white space-y-6">
              <h2 className="text-3xl md:text-4xl font-bold">
                Prêt à récupérer vos week-end ?
              </h2>
              <p className="text-xl text-white/90">
                1 page offerte à l'inscription pour tester.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  size="lg"
                  className="text-lg px-8 py-6 bg-white text-purple-600 hover:bg-purple-50"
                  asChild
                >
                  <Link href="/auth/register">
                    Créer mon compte gratuit
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Link>
                </Button>
              </div>
              <p className="text-sm text-white/70">
                Aucune carte bancaire requise • Annulez quand vous voulez
              </p>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
