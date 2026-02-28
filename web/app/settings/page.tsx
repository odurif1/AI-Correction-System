"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { api } from "@/lib/api";
import { Loader2, Save, CheckCircle2 } from "lucide-react";
import { useState } from "react";

export default function SettingsPage() {
  const router = useRouter();

  // Redirect normal users away from settings (admin only)
  // TODO: Add proper admin check when admin system is implemented
  useEffect(() => {
    router.push("/dashboard");
  }, [router]);
  const [formData, setFormData] = useState({
    llm1_provider: "",
    llm1_model: "",
    llm2_provider: "",
    llm2_model: "",
    comparison_mode: false,
    confidence_auto: 0.85,
    confidence_flag: 0.60,
  });
  const [showSuccess, setShowSuccess] = useState(false);

  // Fetch current settings
  const { data: settings, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: () => api.getSettings(),
  });

  // Update form when settings load
  useEffect(() => {
    if (settings) {
      setFormData({
        llm1_provider: (settings as Record<string, unknown>).llm1_provider as string || "",
        llm1_model: (settings as Record<string, unknown>).llm1_model as string || "",
        llm2_provider: (settings as Record<string, unknown>).llm2_provider as string || "",
        llm2_model: (settings as Record<string, unknown>).llm2_model as string || "",
        comparison_mode: (settings as Record<string, unknown>).comparison_mode as boolean || false,
        confidence_auto: (settings as Record<string, unknown>).confidence_auto as number || 0.85,
        confidence_flag: (settings as Record<string, unknown>).confidence_flag as number || 0.60,
      });
    }
  }, [settings]);

  // Fetch providers
  const { data: providers } = useQuery({
    queryKey: ["providers"],
    queryFn: () => api.listProviders(),
  });

  // Update settings mutation
  const updateMutation = useMutation({
    mutationFn: () =>
      api.updateSettings({
        llm1_provider: formData.llm1_provider || undefined,
        llm1_model: formData.llm1_model || undefined,
        llm2_provider: formData.llm2_provider || undefined,
        llm2_model: formData.llm2_model || undefined,
        comparison_mode: formData.comparison_mode,
        confidence_auto: formData.confidence_auto,
        confidence_flag: formData.confidence_flag,
      }),
    onSuccess: () => {
      setShowSuccess(true);
    },
  });

  // Auto-dismiss success message after 3 seconds
  useEffect(() => {
    if (showSuccess) {
      const timer = setTimeout(() => {
        setShowSuccess(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [showSuccess]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container py-8 max-w-2xl">
        <h1 className="text-3xl font-bold tracking-tight mb-8">Paramètres</h1>

        {/* Provider Status */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Fournisseurs IA</CardTitle>
            <CardDescription>
              État des fournisseurs d'IA configurés
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {providers?.map((provider) => (
                <div
                  key={provider.id}
                  className="flex items-center justify-between p-3 bg-muted rounded-md"
                >
                  <div>
                    <div className="font-medium">{provider.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {provider.models.map((m) => m.name).join(", ")}
                    </div>
                  </div>
                  {provider.configured ? (
                    <div className="flex items-center text-success">
                      <CheckCircle2 className="h-5 w-5 mr-1" />
                      Configuré
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm">
                      Non configuré
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Grading Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Paramètres de correction</CardTitle>
            <CardDescription>
              Configurer les modèles IA et les seuils de notation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* LLM1 Configuration */}
              <div className="space-y-4">
                <h3 className="font-medium">IA principale</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="llm1-provider">Fournisseur</Label>
                    <Select
                      value={formData.llm1_provider}
                      onValueChange={(value) =>
                        setFormData({ ...formData, llm1_provider: value, llm1_model: "" })
                      }
                    >
                      <SelectTrigger id="llm1-provider">
                        <SelectValue placeholder="Choisir un fournisseur" />
                      </SelectTrigger>
                      <SelectContent>
                        {providers?.filter((p) => p.configured).map((provider) => (
                          <SelectItem key={provider.id} value={provider.id}>
                            {provider.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="llm1-model">Modèle</Label>
                    <Select
                      value={formData.llm1_model}
                      onValueChange={(value) =>
                        setFormData({ ...formData, llm1_model: value })
                      }
                      disabled={!formData.llm1_provider}
                    >
                      <SelectTrigger id="llm1-model">
                        <SelectValue placeholder="Choisir un modèle" />
                      </SelectTrigger>
                      <SelectContent>
                        {providers
                          ?.find((p) => p.id === formData.llm1_provider)
                          ?.models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              {model.name}
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              {/* LLM2 Configuration */}
              <div className="space-y-4">
                <h3 className="font-medium">IA secondaire (mode double)</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="llm2-provider">Fournisseur</Label>
                    <Select
                      value={formData.llm2_provider}
                      onValueChange={(value) =>
                        setFormData({ ...formData, llm2_provider: value, llm2_model: "" })
                      }
                    >
                      <SelectTrigger id="llm2-provider">
                        <SelectValue placeholder="Choisir un fournisseur" />
                      </SelectTrigger>
                      <SelectContent>
                        {providers?.filter((p) => p.configured).map((provider) => (
                          <SelectItem key={provider.id} value={provider.id}>
                            {provider.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="llm2-model">Modèle</Label>
                    <Select
                      value={formData.llm2_model}
                      onValueChange={(value) =>
                        setFormData({ ...formData, llm2_model: value })
                      }
                      disabled={!formData.llm2_provider}
                    >
                      <SelectTrigger id="llm2-model">
                        <SelectValue placeholder="Choisir un modèle" />
                      </SelectTrigger>
                      <SelectContent>
                        {providers
                          ?.find((p) => p.id === formData.llm2_provider)
                          ?.models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              {model.name}
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              {/* Thresholds */}
              <div className="space-y-4">
                <h3 className="font-medium">Seuils de confiance</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="confidence-auto">Seuil auto-acceptation</Label>
                    <Input
                      id="confidence-auto"
                      type="number"
                      step={0.05}
                      min={0}
                      max={1}
                      value={formData.confidence_auto}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          confidence_auto: parseFloat(e.target.value) || 0,
                        })
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Notes au-dessus de ce seuil auto-acceptées
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confidence-flag">Seuil de signalement</Label>
                    <Input
                      id="confidence-flag"
                      type="number"
                      step={0.05}
                      min={0}
                      max={1}
                      value={formData.confidence_flag}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          confidence_flag: parseFloat(e.target.value) || 0,
                        })
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Notes en dessous à vérifier
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex justify-end pt-4">
                <Button
                  onClick={() => updateMutation.mutate()}
                  disabled={updateMutation.isPending}
                >
                  {updateMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  Enregistrer
                </Button>
              </div>

              {showSuccess && (
                <div className="text-sm text-success text-center">
                  Paramètres enregistrés
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </main>

      <Footer />
    </div>
  );
}
