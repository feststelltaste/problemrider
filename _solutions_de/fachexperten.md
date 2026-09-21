---
title: Fachexperten
description: Direkte Einbindung von Fachexperten in die Entwicklung.
category:
- Team
- Requirements
problems:
- stakeholder-developer-communication-gap
- requirements-ambiguity
- implicit-knowledge
- knowledge-gaps
- legacy-business-logic-extraction-difficulty
- poor-domain-model
- inappropriate-skillset
layout: solution
lang: de
en_slug: domain-experts
related_solutions:
- slug: domain-modeling
  similarity: 0.75
- slug: domain-patterns
  similarity: 0.7
- slug: on-site-customer
  similarity: 0.7
- slug: subject-matter-reviews
  similarity: 0.7
- slug: code-reviews
  similarity: 0.7
- slug: domain-quiz
  similarity: 0.7
---

## Description

Fachexperten direkt einzubinden bedeutet, Menschen mit tiefem Geschäftswissen in die tägliche Arbeit des Entwicklungsteams einzubetten — Reviews, Pairing-Sitzungen, Walkthroughs —, statt sie als Ressource zu behandeln, die nur über einen formellen Anfragekanal erreichbar ist, wenn zufällig eine Frage aufkommt. Dies ist besonders folgenreich für Legacy-Systeme, wo die Menschen, die die Geschäftsregeln ursprünglich in die Software kodiert haben, häufig die Organisation verlassen haben und Logik hinterlassen, die korrekt läuft, deren Begründung, Randfälle und Annahmen aber niemandem mehr bekannt sind, der das System aktiv pflegt. Ein Fachexperte, der neben Entwicklern arbeitet, kann validieren, ob extrahierte oder neu implementierte Geschäftsregeln tatsächlich vollständig und korrekt sind, und kann das spezifische und kostspielige Fehlermuster erkennen, bei dem eine Legacy-Implementierung getreu eine Regel kodiert, die vor Jahren durch eine regulatorische oder geschäftliche Änderung ersetzt wurde, aber nie im Code aktualisiert wurde. Ihre Anwesenheit schließt auch die Kommunikationslücke zwischen Stakeholdern und Entwicklern in Echtzeit, während Design und Implementierung, statt nachdem ein Feature bereits mit einem eingebackenen Missverständnis ausgeliefert wurde. Weil Expertenzeit knapp ist und die Darstellung eines einzelnen Experten immer noch einen idealisierten statt tatsächlichen Prozess widerspiegeln kann, sollte ihr Wissen in strukturierter, dauerhafter Dokumentation erfasst werden, während es übertragen wird, was das Risiko reduziert, dass kritisches Verständnis erneut in einer Person konzentriert wird, die möglicherweise irgendwann geht.

## How to Apply ◆

- Betten Sie Fachexperten direkt in Entwicklungsteams ein, statt sie nur über formelle Anfragekanäle verfügbar zu machen.
- Planen Sie regelmäßige Sitzungen, in denen Fachexperten Entwickler durch Geschäftsprozesse und im Legacy-System kodierte Regeln führen.
- Lassen Sie Fachexperten an Code-Reviews von Geschäftslogikänderungen teilnehmen, um Korrektheit zu validieren.
- Nutzen Sie Fachexperten, um zu verifizieren, dass extrahierte Legacy-Geschäftsregeln vollständig und akkurat sind, bevor sie neu implementiert werden.
- Schaffen Sie Gelegenheiten für informellen Wissenstransfer: Pair-Programming-Sitzungen, Whiteboard-Diskussionen und Beratung am Arbeitsplatz.
- Dokumentieren Sie von Experten erfasstes Domänenwissen in einem strukturierten Format, um das Bus-Factor-Risiko zu reduzieren.

## Tradeoffs ⇄

**Vorteile:**
- Reduziert Missverständnisse zwischen Geschäftsabsicht und technischer Implementierung.
- Beschleunigt das Verständnis von Legacy-Geschäftslogik, die möglicherweise nicht dokumentiert ist.
- Fängt Geschäftslogikfehler während der Entwicklung ab statt nach dem Deployment.
- Baut Entwicklerempathie für Nutzerbedürfnisse und Geschäftseinschränkungen auf.

**Kosten:**
- Die Zeit von Fachexperten ist wertvoll und oft begrenzt; ihre Einbindung erfordert sorgfältige Planung.
- Experten könnten Schwierigkeiten haben, ihr Wissen in Begriffen auszudrücken, auf denen Entwickler handeln können.
- Übermäßige Abhängigkeit von einem einzelnen Fachexperten schafft einen Wissensengpass.
- Fachexperten könnten idealisierte Prozesse statt tatsächlich implementiertes Verhalten beschreiben.

## How It Could Be

Ein Legacy-Steuerberechnungssystem enthält Hunderte von über zwei Jahrzehnte angehäuften Geschäftsregeln, aber die ursprünglichen Entwickler haben das Unternehmen verlassen. Ein Steuerspezialist wird während eines Modernisierungsprojekts ins Entwicklungsteam eingebettet. Sie identifiziert zahlreiche Fälle, in denen der Legacy-Code Regeln implementiert, die vor Jahren durch regulatorische Änderungen ersetzt wurden, sowie mehrere Randfälle, in denen der Code vom korrekten Steuerrecht abweicht. Ihre Beteiligung verhindert, dass das Team Fehler getreu ins neue System repliziert, und stellt sicher, dass das modernisierte System aktuelle Vorschriften korrekt implementiert. Die Fachexpertin hilft dem Team auch, ein gemeinsames Vokabular für Steuerkonzepte zu etablieren, was Missverständnisse eliminiert, die zuvor zu Wochen von Nacharbeit pro Sprint geführt hatten.
