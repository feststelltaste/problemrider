---
title: Ensemble Programming
description: Lösung komplexer Design- und Debugging-Herausforderungen durch gemeinsame
  Programmierung als Gruppe an einem Arbeitsplatz.
category:
- Team
- Code
problems:
- knowledge-silos
- knowledge-dependency
- difficult-code-comprehension
- debugging-difficulties
- complex-and-obscure-logic
- knowledge-gaps
- team-silos
- slow-knowledge-transfer
- poor-teamwork
- team-dysfunction
- fear-of-conflict
layout: solution
lang: de
en_slug: collaborative-problem-solving
related_solutions:
- slug: pair-and-mob-programming
  similarity: 0.85
- slug: code-reading-sessions
  similarity: 0.7
- slug: structured-onboarding-program
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
- slug: communities-of-practice
  similarity: 0.65
- slug: internal-technical-coaching
  similarity: 0.65
---

## Description

Ensemble Programming, auch bekannt als Mob Programming, bringt eine gesamte kleine Gruppe von Entwicklern an einem gemeinsamen Arbeitsplatz und Bildschirm zusammen, um gleichzeitig an einem einzigen Problem zu arbeiten, wobei eine Person als „Driver" agiert und tippt, während der Rest der Gruppe navigiert, diskutiert und leitet, und die Driver-Rolle alle zehn bis fünfzehn Minuten rotiert. Statt ein Problem über Einzelpersonen aufzuteilen, die jeweils eine Teilansicht davon haben, bündelt die Technik bewusst das Wissen aller im selben Moment, sodass alles, was die Gruppe produziert, das kombinierte Verständnis aller Anwesenden widerspiegelt, statt der begrenzten Perspektive dessen, der zufällig die Aufgabe übernommen hat. Dies eignet sich besonders für die Teile eines Legacy-Systems, die kein einzelner Entwickler vollständig versteht, weil die Verantwortung für den Code über die Jahre durch mehrere verschiedene Teams ohne Dokumentation ging, und sein Verständnis erfordert, Wissensfragmente zusammenzusetzen, die einzeln über verschiedene Personen verstreut sind. Solchen Code als Ensemble durchzuarbeiten verwandelt das, was sonst eine langsame, einsame Reverse-Engineering-Anstrengung wäre — oder ein Bug, der ungelöst bleibt, weil er zwischen Expertisebereiche fällt — in eine Sitzung, in der Menschen mit sich ergänzendem Teilwissen es in Echtzeit kombinieren, oft ein Verständnis erreichend, das keiner von ihnen allein hätte erreichen können. Es ist außerdem ein effektiver, hochbandiger Weg, ein neues Teammitglied direkt in die schwierigsten Bereiche der Legacy-Codebasis einzuführen, es in Kontext eintauchend, der sonst Monate brauchen würde, um unabhängig angehäuft zu werden. Die offensichtlichen Kosten sind, dass es die Zeit mehrerer Entwickler gleichzeitig für ein einziges Arbeitsstück verbraucht, was nach konventionellen Einzelausgabe-Maßstäben ineffizient aussehen kann, und es ist am besten für genuin schwierige, folgenreiche Probleme reserviert, statt auf Routinearbeit angewendet zu werden, die ein einzelner Entwickler ebenso gut allein handhaben würde.

## How to Apply ◆

> In Legacy-Systemen ist Ensemble Programming (Mob Programming) besonders effektiv, um die komplexesten und am schlechtesten verstandenen Teile der Codebasis anzugehen, wo niemand vollständiges Wissen hat.

- Versammeln Sie das Team (drei bis sechs Personen) an einem Arbeitsplatz mit einem gemeinsamen Bildschirm, und rotieren Sie die „Driver"-Rolle (Person an der Tastatur) alle 10-15 Minuten, während der Rest der Gruppe navigiert und diskutiert.
- Nutzen Sie Ensemble-Sitzungen spezifisch für die anspruchsvollsten Legacy-Code-Aufgaben: Verstehen undokumentierter Geschäftslogik, Debugging intermittierender Produktionsprobleme oder das Design von Migrationsstrategien für eng gekoppelte Komponenten.
- Beziehen Sie Entwickler mit unterschiedlichen Bereichen von Legacy-Systemwissen in dieselbe Sitzung ein, um Teilverständnisse zu vollständigem Verständnis zu kombinieren.
- Etablieren Sie Grundregeln: Alle Entscheidungen gehen durch die Hände des Drivers, und die Gruppe muss ihre Absicht klar genug erklären, damit der Driver sie umsetzen kann.
- Planen Sie Ensemble-Sitzungen für fokussierte Blöcke (zwei bis vier Stunden) mit Pausen, statt ganztägige Sitzungen zu versuchen, die zu Ermüdung führen.
- Nutzen Sie Ensemble Programming für Wissenstransfer beim Onboarding neuer Teammitglieder in komplexe Legacy-Code-Bereiche.

## Tradeoffs ⇄

> Ensemble Programming beschleunigt das Lernen und produziert höherwertige Lösungen für komplexe Probleme, nutzt aber die Zeit mehrerer Entwickler gleichzeitig.

**Vorteile:**

- Kombiniert fragmentiertes Wissen des Legacy-Systems von mehreren Entwicklern und produziert Verständnis, das kein Einzelner allein erreichen könnte.
- Eliminiert Wissenssilos, indem sichergestellt wird, dass mehrere Teammitglieder jedes in Ensemble-Sitzungen produzierte Codestück verstehen.
- Produziert höherwertige Lösungen für komplexe Probleme, weil mehrere Perspektiven Probleme abfangen und Verbesserungen in Echtzeit vorschlagen.
- Beschleunigt Onboarding, indem neue Teammitglieder zusammen mit erfahrenen Entwicklern in die Codebasis eintauchen.

**Kosten und Risiken:**

- Nutzt die Zeit mehrerer Entwickler gleichzeitig, was Managern, die Produktivität nach individueller Ausgabe messen, verschwenderisch erscheinen könnte.
- Ensemble-Sitzungen können von starken Persönlichkeiten dominiert werden, wenn Moderation nicht gemanagt wird, was den Nutzen vielfältiger Perspektiven verringert.
- Ermüdung durch anhaltenden Gruppenfokus kann die Effektivität in langen Sitzungen verringern.
- Nicht alle Aufgaben profitieren von Ensemble Programming — Routinearbeit wird oft effizienter individuell erledigt.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie Ensemble Programming Verständnis von Legacy-Code freischaltet.

Ein Zahlungsverarbeitungsunternehmen hatte ein kritisches Transaktionsabgleichsmodul, das kein einzelner Entwickler vollständig verstand — es war von drei verschiedenen Teams über acht Jahre gebaut worden, wobei jedes Team Schichten ohne Dokumentation hinzufügte. Als ein Abgleichsbug 200.000 $ an Transaktionen betraf, versammelte das Team ein Ensemble von fünf Entwicklern: zwei, die die ursprüngliche Abgleichslogik verstanden, einen, der die Ausnahmebehandlungsschicht gebaut hatte, einen, der das Datenbankschema kannte, und ein neues Teammitglied, das klärende Fragen stellte. Über zwei vierstündige Sitzungen führte das Ensemble den Bug auf eine Race Condition zwischen zwei Abgleichsprozessen zurück, die eingeführt worden war, als das zweite Team Batch-Verarbeitung hinzufügte. Der Fix erforderte die Koordination von Änderungen über drei Module — Änderungen, die jeder einzelne Entwickler Wochen gebraucht hätte, um gut genug zu verstehen, um sie sicher zu implementieren. Das Ensemble dokumentierte außerdem den Abgleichsfluss zum ersten Mal, ein Artefakt schaffend, das die Sitzungen überdauerte.
