"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { Github, Code, CheckCircle2 } from "lucide-react";

export default function ApproachPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1">
        {/* Hero */}
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-50 via-blue-50 to-purple-50 dark:from-purple-950/20 dark:via-blue-950/20 dark:to-purple-950/20 animated-gradient" />

          <div className="relative container py-16 md:py-24">
            <div className="max-w-3xl mx-auto text-center space-y-4">
              <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight">
                Notre approche
              </h1>
              <p className="text-lg text-muted-foreground">
                Pas de magie. Juste de la technologie de pointe, utilisée intelligemment.
              </p>
            </div>
          </div>
        </section>

        {/* Technology Explained */}
        <section className="container py-16 md:py-24">
          <div className="max-w-5xl mx-auto">
            <div className="grid md:grid-cols-2 gap-12 items-center mb-16">
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center shrink-0">
                    <span className="text-purple-600 dark:text-purple-400 font-bold">1</span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg mb-1">L'IA lit votre copie</h3>
                    <p className="text-muted-foreground">
                      Les modèles <strong>vision-language</strong> analysent directement l'image de la copie.
                      Ils lisent l'écriture manuscrite, comprennent les schémas, les calculs, les dessins,
                      le contexte, les barèmes.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center shrink-0">
                    <span className="text-purple-600 dark:text-purple-400 font-bold">2</span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg mb-1">Deux IA corrigent en parallèle</h3>
                    <p className="text-muted-foreground">
                      <strong>Deux modèles différents</strong> notent indépendamment chaque réponse.
                      S'ils sont d'accord, la note est validée. S'ils divergent, les IA se confrontent :
                      chacune revoit sa position à la lumière de l'analyse de l'autre.
                      Vous gardez le dernier mot.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center shrink-0">
                    <span className="text-purple-600 dark:text-purple-400 font-bold">3</span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg mb-1">Annotation comme un vrai prof</h3>
                    <p className="text-muted-foreground">
                      Chaque copie est <strong>annotée</strong> avec des commentaires personnalisés,
                      des soulignements, des corrections en marge. Exactement comme vous le feriez
                      avec votre stylo rouge.
                    </p>
                  </div>
                </div>
              </div>

              <div className="relative">
                <div className="bg-gradient-to-br from-purple-100 to-blue-100 dark:from-purple-950/50 dark:to-blue-950/50 rounded-2xl p-8 space-y-4">
                  <div className="bg-white dark:bg-gray-900 rounded-lg p-4 shadow-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 rounded-full bg-red-400"></div>
                      <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                      <div className="w-3 h-3 rounded-full bg-green-400"></div>
                    </div>
                    <div className="space-y-2 text-sm font-mono">
                      <div className="flex items-center gap-2">
                        <span className="text-green-500">✓</span>
                        <span>LLM1 : 4/5 points</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-green-500">✓</span>
                        <span>LLM2 : 4/5 points</span>
                      </div>
                      <div className="border-t pt-2 mt-2">
                        <span className="text-purple-600 font-semibold">→ Consensus : 4/5 ✓</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-900 rounded-lg p-4 shadow-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 rounded-full bg-red-400"></div>
                      <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                      <div className="w-3 h-3 rounded-full bg-green-400"></div>
                    </div>
                    <div className="space-y-2 text-sm font-mono">
                      <div className="flex items-center gap-2">
                        <span className="text-yellow-500">⚠</span>
                        <span>LLM1 : 2/5 points</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-yellow-500">⚠</span>
                        <span>LLM2 : 4/5 points</span>
                      </div>
                      <div className="border-t pt-2 mt-2">
                        <span className="text-yellow-600 font-semibold">→ Désaccord → À vérifier</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Key benefits */}
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center p-6">
                <div className="text-3xl font-bold text-purple-600 mb-2">95%</div>
                <div className="text-sm text-muted-foreground">
                  de précision sur les notes finales
                </div>
              </div>
              <div className="text-center p-6 border-x">
                <div className="text-3xl font-bold text-purple-600 mb-2">2x</div>
                <div className="text-sm text-muted-foreground">
                  moins d'erreurs grâce à la double correction
                </div>
              </div>
              <div className="text-center p-6">
                <div className="text-3xl font-bold text-purple-600 mb-2">100%</div>
                <div className="text-sm text-muted-foreground">
                  de contrôle conservé par le professeur
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Open Source Section */}
        <section className="bg-muted/30">
          <div className="container py-16 md:py-24">
            <div className="max-w-3xl mx-auto text-center space-y-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-900 dark:bg-white mb-4">
                <Github className="h-8 w-8 text-white dark:text-gray-900" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold">
                Open Source
              </h2>
              <p className="text-lg text-muted-foreground">
                Un projet open source <strong>créé par un ancien prof</strong>,
                qui comprend les vrais besoins de la correction.
                Code auditable, auto-hébergeable, transparent.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button variant="outline" size="lg" asChild>
                  <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                    <Github className="mr-2 h-5 w-5" />
                    Voir sur GitHub
                  </a>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                    <Code className="mr-2 h-5 w-5" />
                    Contribuer
                  </a>
                </Button>
              </div>
              <div className="flex flex-wrap justify-center gap-6 text-sm text-muted-foreground pt-4">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Code auditable
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Licence MIT
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Auto-hébergeable
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-600 to-blue-600 animated-gradient" />

          <div className="relative container py-16">
            <div className="max-w-2xl mx-auto text-center text-white space-y-6">
              <h2 className="text-2xl md:text-3xl font-bold">
                Convaincu ?
              </h2>
              <p className="text-white/90">
                Testez gratuitement avec 1 page.
              </p>
              <Button size="lg" variant="secondary" asChild>
                <Link href="/auth/register">
                  Créer mon compte
                </Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
