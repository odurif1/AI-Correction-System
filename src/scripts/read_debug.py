#!/usr/bin/env python3
"""
Lecteur de logs de debug - Format humain.

Usage:
    python -m scripts.read_debug <session_id>
    python -m scripts.read_debug <session_id> --call 2
    python -m scripts.read_debug <session_id> --provider GeminiProvider
    python -m scripts.read_debug <session_id> --summary
"""

import json
import sys
from pathlib import Path
from typing import Optional
import argparse


def find_debug_log(session_id: str) -> Optional[Path]:
    """Trouve le fichier debug_log.json pour une session."""
    base_path = Path(__file__).parent.parent.parent / "data" / session_id / "debug" / "debug_log.json"
    if base_path.exists():
        return base_path

    # Essayer dans outputs/
    alt_path = Path(__file__).parent.parent.parent / "outputs" / session_id / "debug" / "debug_log.json"
    if alt_path.exists():
        return alt_path

    return None


def print_summary(data: dict):
    """Affiche un résumé des appels."""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)
    print(f"Temps de capture: {data.get('capture_time', 'N/A')}")
    print(f"Nombre total d'appels: {data.get('total_calls', 0)}")

    calls = data.get('calls', [])

    # Stats par provider
    providers = {}
    for call in calls:
        p = call.get('provider', 'Unknown')
        providers[p] = providers.get(p, 0) + 1

    print("\nAppels par provider:")
    for p, count in providers.items():
        print(f"  - {p}: {count}")

    # Stats par type
    types = {}
    for call in calls:
        t = call.get('call_type', 'Unknown')
        types[t] = types.get(t, 0) + 1

    print("\nAppels par type:")
    for t, count in types.items():
        print(f"  - {t}: {count}")

    # Tokens totaux
    total_prompt = sum(c.get('tokens', {}).get('prompt', 0) for c in calls)
    total_completion = sum(c.get('tokens', {}).get('completion', 0) for c in calls)
    print(f"\nTokens totaux:")
    print(f"  - Prompt: {total_prompt:,}")
    print(f"  - Completion: {total_completion:,}")
    print(f"  - Total: {total_prompt + total_completion:,}")

    # Erreurs
    errors = sum(1 for c in calls if c.get('error'))
    if errors:
        print(f"\n⚠️  Erreurs: {errors}")


def print_call(call: dict, index: int, show_prompt: bool = True, show_response: bool = True, llm_num: int = None, call_num: int = None):
    """Affiche un appel API de manière lisible."""
    print("\n" + "="*60)
    if llm_num is not None and call_num is not None:
        print(f"📞 APPEL #{llm_num}.{call_num}")
    else:
        print(f"📞 APPEL #{index + 1}")
    print("="*60)

    print(f"Provider: {call.get('provider', 'N/A')}")
    print(f"Model: {call.get('model', 'N/A')}")
    print(f"Type: {call.get('call_type', 'N/A')}")
    print(f"Durée: {call.get('duration_ms', 0):.0f}ms")

    tokens = call.get('tokens', {})
    if tokens:
        print(f"Tokens: {tokens.get('prompt', 0)} prompt + {tokens.get('completion', 0)} completion")

    if call.get('error'):
        print(f"\n❌ ERREUR: {call['error']}")

    if call.get('images'):
        print(f"\n📷 Images ({len(call['images'])}):")
        for img in call['images'][:3]:  # Max 3
            print(f"  - {Path(img).name}")
        if len(call['images']) > 3:
            print(f"  ... et {len(call['images']) - 3} autres")

    if show_prompt and call.get('prompt'):
        prompt = call['prompt']
        print(f"\n📝 PROMPT ({len(prompt)} chars):")
        print("-"*40)
        print(prompt)

    if show_response and call.get('response'):
        response = call['response']
        print(f"\n📤 RÉPONSE ({len(response)} chars):")
        print("-"*40)
        print(response)


def main():
    parser = argparse.ArgumentParser(description="Lecteur de logs de debug")
    parser.add_argument("session_id", help="ID de la session")
    parser.add_argument("--call", "-c", type=int, help="Afficher un appel spécifique (0-indexed)")
    parser.add_argument("--provider", "-p", help="Filtrer par provider")
    parser.add_argument("--type", "-t", help="Filtrer par type (vision/text)")
    parser.add_argument("--summary", "-s", action="store_true", help="Afficher seulement le résumé")
    parser.add_argument("--no-prompt", action="store_true", help="Ne pas afficher les prompts")
    parser.add_argument("--no-response", action="store_true", help="Ne pas afficher les réponses")
    parser.add_argument("--list", "-l", action="store_true", help="Lister les appels sans détails")

    args = parser.parse_args()

    # Trouver le fichier
    debug_path = find_debug_log(args.session_id)
    if not debug_path:
        print(f"❌ Fichier debug non trouvé pour la session: {args.session_id}")
        print("\nCherché dans:")
        print(f"  - data/{args.session_id}/debug/debug_log.json")
        print(f"  - outputs/{args.session_id}/debug/debug_log.json")
        sys.exit(1)

    print(f"📄 Fichier: {debug_path}")

    # Charger les données
    with open(debug_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    calls = data.get('calls', [])

    # Filtrer
    if args.provider:
        calls = [c for c in calls if args.provider.lower() in c.get('provider', '').lower()]
    if args.type:
        calls = [c for c in calls if args.type.lower() in c.get('call_type', '').lower()]

    # Calculer les numéros Y.X et trier par vague (X) puis LLM (Y)
    llm_nums = _compute_parallel_groups(calls)

    # Créer liste avec indices et trier par (llm_num, call_num) - grouper par LLM
    indexed_calls = list(enumerate(calls))
    indexed_calls.sort(key=lambda x: (llm_nums[x[0]][0], llm_nums[x[0]][1]))

    # Afficher
    if args.summary:
        print_summary(data)
    elif args.list:
        print("\n📋 LISTE DES APPELS:")
        for orig_idx, call in indexed_calls:
            lnum, cnum = llm_nums[orig_idx]
            print(f"  [{lnum}.{cnum}] {call.get('provider', '?'):20} | {call.get('call_type', '?'):8} | {call.get('duration_ms', 0):6.0f}ms | {call.get('model', '?')}")
    elif args.call is not None:
        if 0 <= args.call < len(calls):
            lnum, cnum = llm_nums[args.call]
            print_call(calls[args.call], args.call,
                      show_prompt=not args.no_prompt,
                      show_response=not args.no_response,
                      llm_num=lnum, call_num=cnum)
        else:
            print(f"❌ Appel {args.call} non trouvé (0-{len(calls)-1})")
    else:
        # Afficher le résumé puis tous les appels (triés par vague)
        print_summary(data)
        for orig_idx, call in indexed_calls:
            lnum, cnum = llm_nums[orig_idx]
            print_call(call, orig_idx,
                      show_prompt=not args.no_prompt,
                      show_response=not args.no_response,
                      llm_num=lnum, call_num=cnum)


def _detect_phase(call: dict) -> int:
    """
    Détecte la phase d'un appel basée sur son contenu.

    Returns:
        1 = batch initial, 2 = vérification, 3 = ultimatum
    """
    prompt = call.get('prompt', '').lower()
    call_type = call.get('call_type', '')

    # Phase 3: Ultimatum
    if 'ultimatum' in prompt or 'décision finale' in prompt:
        return 3

    # Phase 2: Vérification
    if 'désaccord' in prompt or 'réexamine' in prompt or call_type == 'chat':
        return 2

    # Phase 1: Batch initial (vision calls without verification keywords)
    return 1


def _compute_parallel_groups(calls: list, tolerance_seconds: float = 2.0) -> list:
    """
    Calcule les numéros d'appels: (numéro_llm, numéro_phase).

    X = LLM (1 ou 2) - extrait de llm_id dans le log, ou deviné par timestamp
    Y = Phase (1=batch, 2=vérification, 3=ultimatum)

    Returns:
        Liste de tuples (llm_number, phase_number) pour chaque appel
    """
    if not calls:
        return []

    result = []

    # D'abord, vérifier si llm_id est disponible dans les logs
    has_llm_id = any(call.get('llm_id', 0) > 0 for call in calls)

    if has_llm_id:
        # Utiliser llm_id du log
        for call in calls:
            llm_id = call.get('llm_id', 1)
            phase = _detect_phase(call)
            result.append((llm_id, phase))
    else:
        # Fallback: deviner par timestamp (ancien comportement)
        from datetime import datetime

        phase_groups = {}
        for idx, call in enumerate(calls):
            phase = _detect_phase(call)
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append((idx, call))

        result = [None] * len(calls)

        for phase in sorted(phase_groups.keys()):
            phase_calls = phase_groups[phase]

            def get_timestamp(item):
                ts = item[1].get('timestamp', '')
                try:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    return datetime.min

            phase_calls.sort(key=get_timestamp)

            groups = []
            current_group = []

            for item in phase_calls:
                if not current_group:
                    current_group = [item]
                else:
                    t1 = get_timestamp(current_group[0])
                    t2 = get_timestamp(item)
                    diff = abs((t2 - t1).total_seconds())

                    if diff < tolerance_seconds:
                        current_group.append(item)
                    else:
                        groups.append(current_group)
                        current_group = [item]

            if current_group:
                groups.append(current_group)

            for group in groups:
                if len(group) == 1:
                    idx = group[0][0]
                    result[idx] = (1, phase)
                else:
                    for i, (idx, call) in enumerate(sorted(group, key=lambda x: x[0])):
                        result[idx] = (i + 1, phase)

        for idx in range(len(result)):
            if result[idx] is None:
                result[idx] = (1, _detect_phase(calls[idx]))

    return result


if __name__ == "__main__":
    main()
