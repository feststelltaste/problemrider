---
title: Dark Launches
description: Begrenzung der Auswirkungsreichweite neuer Features durch verstecktes
  Deployment für eine Teilmenge der Nutzer.
category:
- Operations
- Process
problems:
- deployment-risk
- large-risky-releases
- release-anxiety
- fear-of-change
- release-instability
- high-defect-rate-in-production
layout: solution
lang: de
en_slug: dark-launches
related_solutions:
- slug: canary-releases
  similarity: 0.8
- slug: feature-toggles
  similarity: 0.75
- slug: chaos-engineering
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.7
- slug: restore-points
  similarity: 0.7
- slug: error-budgets
  similarity: 0.7
---

## Description

Dark Launches deployen neuen Code in einem deaktivierten oder versteckten Zustand in Produktion und aktivieren ihn dann selektiv — für interne Nutzer, eine kleine Testgruppe oder über Schattentraffic, der den neuen Pfad ausübt, ohne zu beeinflussen, was Nutzer tatsächlich sehen —, sodass die neue Funktionalität gegen echte Produktionsbedingungen validiert wird, bevor sie breit ausgesetzt wird. Legacy-Systeme tragen oft eine erhöhte Angst vor großen Releases, gerade weil vergangene Big-Bang-Rollouts schiefgegangen sind, und diese Angst treibt Releases wiederum dazu, noch größer und weiter auseinander zu werden, da niemand die Erfahrung eines Releases wiederholen will, das zu viel auf einmal ohne Möglichkeit zu isolieren, was gebrochen ist, geändert hat. Indem Deployment über Feature Flags von der nutzersichtbaren Freigabe entkoppelt wird, erlauben Dark Launches einem Team, kontinuierlich Code auszuliefern, während die Exposition unabhängig kontrolliert wird, sodass Probleme abgefangen und behoben — oder das Feature sofort ausgeschaltet — werden können, bevor die meisten Nutzer je betroffen sind. Derselbe Mechanismus unterstützt das parallele Betreiben einer alten und neuen Implementierung in Produktion, wobei beide echte Eingaben erhalten und Ausgaben automatisch verglichen werden, was besonders wertvoll ist, wenn ein kritisches Stück Legacy-Infrastruktur unter strengen regulatorischen oder Zuverlässigkeitseinschränkungen ersetzt wird, die einen direkten Umstieg ausschließen. Die Kosten dieser Sicherheit sind zusätzliche Komplexität: Die Feature-Flag-Infrastruktur selbst muss gebaut und gepflegt werden, dark-gelaunchter Code läuft weiterhin und verbraucht Ressourcen in Produktion, selbst während er versteckt ist, und Flags, die nach einem vollständigen Rollout nicht aufgeräumt werden, häufen sich als eigene Form technischer Schuld an.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Implementieren Sie eine Feature-Flag-Infrastruktur, die Features ohne Neu-Deployment aktivieren oder deaktivieren kann
- Deployen Sie neuen Code in Produktion in deaktiviertem Zustand und aktivieren Sie ihn dann selektiv für interne Nutzer oder eine kleine Testgruppe
- Nutzen Sie Schattentraffic, um neue Codepfade mit echten Produktionsdaten auszuüben, ohne nutzersichtbare Antworten zu beeinflussen
- Überwachen Sie die Performance und Korrektheit dark-gelaunchter Features durch dedizierte Kennzahlen und Logging
- Erweitern Sie die Nutzergruppe schrittweise, während das Vertrauen wächst, mittels prozentbasierter Rollouts
- Etablieren Sie Kill-Switch-Prozeduren, die ein dark-gelaunchtes Feature sofort deaktivieren können, wenn Probleme erkannt werden
- Räumen Sie Feature Flags auf, sobald ein Feature vollständig ausgerollt ist, um Flag-Anhäufung zu vermeiden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Validiert neue Features mit echtem Produktionstraffic und -daten, ohne alle Nutzer einem Risiko auszusetzen
- Entkoppelt Deployment von Feature-Freigabe, was unabhängige Taktungen ermöglicht
- Bietet einen schnellen Rollback-Mechanismus durch Umschalten von Feature Flags
- Reduziert Angst rund um große Feature-Einführungen, indem schrittweise Validierung erlaubt wird

**Kosten und Risiken:**
- Feature-Flag-Infrastruktur fügt der Codebasis und dem Deployment-Prozess Komplexität hinzu
- Angehäufte Feature Flags erzeugen technische Schuld, wenn sie nach vollständigem Rollout nicht aufgeräumt werden
- Dark-gelaunchter Code wird weiterhin in Produktion ausgeführt und kann Performance beeinträchtigen oder Nebeneffekte verursachen
- Schattentraffic-Ansätze erfordern sorgfältige Handhabung, um unbeabsichtigte Schreibvorgänge oder Zustandsänderungen zu vermeiden
- Das Testen wird mit mehreren Feature-Flag-Kombinationen komplexer

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Bankanwendung musste ihre Transaktionsverarbeitungs-Engine ersetzen, konnte aber wegen regulatorischer Anforderungen keinen Big-Bang-Umstieg riskieren. Das Team deployte die neue Engine neben der alten und nutzte Dark Launching, um beide Engines parallel laufen zu lassen. Echte Transaktionen wurden von der alten Engine verarbeitet, während die neue Engine Schattenkopien erhielt und sie unabhängig verarbeitete. Ergebnisse wurden automatisch verglichen, und Diskrepanzen wurden zur Untersuchung protokolliert. Über acht Wochen löste das Team 12 Randfälle, die Testing nicht aufgedeckt hatte. Der finale Umstieg war ein einfacher Feature-Flag-Schalter, der Sekunden zur Ausführung brauchte und ebenso schnell rückgängig gemacht werden konnte.
