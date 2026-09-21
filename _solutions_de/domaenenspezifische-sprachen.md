---
title: Domänenspezifische Sprachen
description: Nutzung von Programmiersprachen, die speziell an die Domäne angepasst
  sind, für fachliche Ausdrücke und Regeln.
category:
- Code
- Architecture
problems:
- complex-and-obscure-logic
- legacy-business-logic-extraction-difficulty
- difficult-code-comprehension
- poor-domain-model
- stakeholder-developer-communication-gap
- requirements-ambiguity
layout: solution
lang: de
en_slug: domain-specific-languages
related_solutions:
- slug: rule-based-systems
  similarity: 0.7
- slug: ubiquitous-language
  similarity: 0.7
- slug: decision-tables
  similarity: 0.7
- slug: domain-experts
  similarity: 0.7
- slug: domain-patterns
  similarity: 0.7
- slug: domain-modeling
  similarity: 0.65
---

## Description

Eine domänenspezifische Sprache erlaubt es, Geschäftsregeln — Preislogik, Validierungsregeln, Workflow-Definitionen — direkt in Vokabular und Struktur auszudrücken, die der Domäne nativ sind, entweder als interne fließende API, die über die bestehende Allzwecksprache gelegt wird, oder als externe Sprache mit eigener benutzerdefinierter Syntax und eigenem Parser, statt in Allzweckcode vergraben zu sein, den nur Entwickler lesen können. Legacy-Systeme drücken genau diese Art von Regeln routinemäßig als tief verschachtelte Bedingungen aus, verstreut über Tausende Zeilen gewöhnlichen Anwendungscodes, was es effektiv unmöglich macht für die Business-Analysten oder Aktuare, die die Regeln tatsächlich besitzen, zu verifizieren, dass der Code ihre Absicht korrekt implementiert. Solche Logik in eine DSL zu migrieren — schrittweise und bei jedem Schritt gegen bestehendes Verhalten validiert — schließt diese Verifikationslücke direkt, da Domänenexperten die Regeln dann selbst lesen und manchmal sogar verfassen können, ohne über einen Entwicklermittler zu gehen. Dies entkoppelt auch den Ausdruck einer Geschäftsregel von ihrer technischen Implementierung, was bedeutet, dass wenn sich eine Regel ändert — wie Preismodelle und regulatorische Anforderungen es häufig tun —, die Änderung in Domänenbegriffen vorgenommen werden kann und einen Bruchteil der Zeit braucht, die das Modifizieren und erneute Testen von Allzweckcode einst erforderte. Die Vorabkosten sind echt: Eine DSL gut zu gestalten erfordert spezialisierte Fähigkeiten, und eine schlecht gestaltete wird schwerer verständlich als der Code, den sie ersetzte, sodass dieser Ansatz am besten für Domänen reserviert bleibt, in denen Geschäftsregeln sowohl komplex als auch häufigem Wandel unterworfen sind.

## How to Apply ◆

- Identifizieren Sie Bereiche der Legacy-Codebasis, in denen Geschäftsregeln in Allzweckcode ausgedrückt sind, der für Nicht-Entwickler schwer zu verstehen ist.
- Gestalten Sie eine domänenspezifische Sprache (DSL), die es erlaubt, Geschäftsregeln in domänennatürlichen Begriffen auszudrücken (z. B. Preisregeln, Validierungsregeln, Workflow-Definitionen).
- Implementieren Sie die DSL mittels eines internen Ansatzes (fließende API innerhalb der bestehenden Programmiersprache) oder eines externen Ansatzes (benutzerdefinierte Syntax mit einem Parser).
- Migrieren Sie Geschäftsregeln schrittweise vom Legacy-Code in die DSL, wobei jede Migration gegen das bestehende Verhalten validiert wird.
- Ermöglichen Sie Domänenexperten, in der DSL ausgedrückte Regeln zu lesen und zu überprüfen, selbst wenn sie sie nicht direkt verfassen können.
- Bieten Sie Tooling-Unterstützung: Syntax-Hervorhebung, Validierung und Testfähigkeiten für die DSL.

## Tradeoffs ⇄

**Vorteile:**
- Macht Geschäftsregeln für Domänenexperten lesbar, was direkte Validierung der Implementierungskorrektheit ermöglicht.
- Trennt den Ausdruck von Geschäftsregeln von technischen Implementierungsbelangen.
- Reduziert den Aufwand, Geschäftsregeln zu ändern, wenn sie sich ändern, da Änderungen in Domänenbegriffen ausgedrückt werden.
- Kann das Codevolumen, das zum Ausdruck komplexer Geschäftslogik nötig ist, erheblich reduzieren.

**Kosten:**
- Die Gestaltung und Implementierung einer DSL erfordert spezialisierte Fähigkeiten und erhebliche Vorabinvestition.
- Entwickler müssen die DSL zusätzlich zur Allzwecksprache lernen.
- Schlecht gestaltete DSLs können schwerer verständlich sein als der Allzweckcode, den sie ersetzen.
- DSLs erfordern Pflege: die Sprache selbst, ihr Tooling und ihre Dokumentation.
- Das Debuggen von in DSL ausgedrückter Logik kann herausfordernd sein, wenn Fehlermeldungen schlecht auf die Domänensprache abbilden.

## How It Could Be

Ein Legacy-Versicherungsunternehmen hat Prämienberechnungsregeln, eingebettet in Tausende Zeilen Java-Code mit tief verschachtelten Bedingungen. Business-Analysten können nicht verifizieren, ob der Code ihre Preismodelle korrekt implementiert. Das Team erstellt eine interne DSL mittels einer fließenden API, die sich wie natürliche Sprache liest: `when(driver.age().isBelow(25)).and(vehicle.type().is("sports")).then(applyMultiplier(1.8))`. Die bestehenden Java-Regeln werden eine nach der anderen zu DSL-Ausdrücken migriert, mit Tests, die äquivalentes Verhalten verifizieren. Aktuare können jetzt die Preisregeln direkt lesen und Fehler erkennen. Als eine regulatorische Änderung neue Preisfaktoren erfordert, wird die Änderung in der DSL ausgedrückt und in Stunden implementiert statt in den Wochen, die zuvor nötig waren, um den Legacy-Java-Code zu modifizieren und zu testen.
