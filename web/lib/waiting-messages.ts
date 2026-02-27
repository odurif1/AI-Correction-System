"use client";

import { useState, useEffect } from "react";

export const WAITING_MESSAGES = [
  "Restez zen...",
  "Prenez un café ☕",
  "L'IA réfléchit intensément 🤔",
  "Patience est mère de sûreté...",
  "La correction arrive bientôt",
  "Un instant, svp...",
  "L'IA est sur le coup !",
  "Ça arrive, ça arrive...",
  "Merci de votre patience 🙏",
  "Presque terminé...",
  "L'IA fait chauffer les neurones",
  "Détendez-vous, on s'occupe de tout",
  "La magie de l'IA en action ✨",
  "Encouragez les algorithmes !",
  "On y est presque...",
  "Corriger est un art délicat",
  "Vos copies sont entre bonnes mains",
  "L'IA travaille dur pour vous",
  "Patientez encore un peu...",
];

export function getRandomMessage(): string {
  return WAITING_MESSAGES[Math.floor(Math.random() * WAITING_MESSAGES.length)];
}

// Hook to rotate messages periodically
export function useRotatingMessage(interval: number = 5000) {
  const [message, setMessage] = useState(getRandomMessage());

  useEffect(() => {
    const timer = setInterval(() => {
      setMessage(getRandomMessage());
    }, interval);

    return () => clearInterval(timer);
  }, [interval]);

  return message;
}
