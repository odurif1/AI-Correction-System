"use client";

import { useState, useEffect } from "react";

export const WAITING_MESSAGES = [
  "Restez zen 🏯",
  "Prenez un café ☕",
  "L'IA réfléchit intensément 🤔",
  "Patience est mère de sûreté...",
  "La patience est la mère de toutes les vertus",
  "Le génie est une longue patience",
  "Mieux vaut bonne attente que mauvaise hâte",
  "La patience est une fleur qui ne se fane jamais 🌸",
  "Attendre et espérer !",
  "La correction arrive bientôt",
  "L'IA est sur le coup !",
  "Ça arrive, ça arrive...",
  "Merci de votre patience 🙏",
  "L'IA fait chauffer ses neurones",
  "Détendez-vous, on s'occupe de tout",
  "La magie de l'IA en action ✨",
  "Encouragez les algorithmes !",
  "Corriger est un art délicat",
  "Vos copies sont entre bonnes mains",
  "L'IA travaille dur pour vous",
];

export function getRandomMessage(): string {
  return WAITING_MESSAGES[Math.floor(Math.random() * WAITING_MESSAGES.length)];
}

// Hook to rotate messages periodically
export function useRotatingMessage(interval: number = 20000) {
  const [message, setMessage] = useState(getRandomMessage());

  useEffect(() => {
    const timer = setInterval(() => {
      setMessage(getRandomMessage());
    }, interval);

    return () => clearInterval(timer);
  }, [interval]);

  return message;
}
