---
title: Wartungs-Overhead
description: Ein unverhältnismäßig hoher Zeit- und Arbeitsaufwand wird für die Wartung
  des bestehenden Systems aufgewendet, oft aufgrund duplizierten Codes und fehlender
  wiederverwendbarer Komponenten.
category:
- Code
- Process
related_problems:
- slug: high-maintenance-costs
  similarity: 0.8
- slug: operational-overhead
  similarity: 0.75
- slug: high-technical-debt
  similarity: 0.7
- slug: maintenance-cost-increase
  similarity: 0.7
- slug: context-switching-overhead
  similarity: 0.65
- slug: increased-cognitive-load
  similarity: 0.65
solutions:
- technical-debt-backlog
- api-deprecation-policy
- api-versioning-strategy
- aspect-oriented-programming-aop
- code-generation
- design-tokens
- rule-based-systems
- standard-software
- dependency-injection-container
- deprecation-strategy
- style-guide
layout: problem
lang: de
en_slug: maintenance-overhead
---

## Description
Wartungs-Overhead ist der exzessive Aufwand, der erforderlich ist, um ein Softwaresystem betriebsbereit und aktuell zu halten. Dies ist ein häufiges Problem in Legacy-Systemen, wo Jahre angehäufter technischer Schulden und Design-Kompromisse die Codebasis schwer bearbeitbar gemacht haben. Wenn der Wartungs-Overhead hoch ist, ist das Entwicklungsteam gezwungen, den Großteil seiner Zeit mit nicht-produktiven Aufgaben zu verbringen, wie dem Beheben von Fehlern, dem Anwenden von Sicherheitspatches und dem Vornehmen kleiner Anpassungen an bestehender Funktionalität. Dies lässt wenig Zeit für Innovation und neue Feature-Entwicklung, was erhebliche Auswirkungen auf das Geschäft haben kann.

## Indicators ⟡
- Das Backlog des Teams wird von Wartungsaufgaben dominiert.
- Es dauert lange, selbst einfache Änderungen am System vorzunehmen.
- Das Team wechselt ständig zwischen verschiedenen Wartungsaufgaben.
- Es gibt eine hohe Rate an Regressionsfehlern, bei denen eine Änderung an einem Teil des Systems etwas anderes bricht.

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Wenn der Großteil der Entwicklerzeit in Wartungsaufgaben fließt, bleibt wenig Kapazität für produktive neue Entwicklung.
- [Unfähigkeit zu innovieren](unfaehigkeit-zu-innovieren.md)
<br/>  Von Wartungsarbeit aufgezehrte Teams können keine Zeit der Erkundung neuer Ansätze oder dem Aufbau neuer Fähigkeiten widmen.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Die meiste Zeit mit repetitiven Wartungsaufgaben statt kreativer Entwicklungsarbeit zu verbringen, demoralisiert Entwickler.
- [Anstieg der Wartungskosten](anstieg-der-wartungskosten.md)
<br/>  Hoher Wartungs-Overhead übersetzt sich direkt in steigende Kosten, während mehr Entwicklerzeit von der Instandhaltung verbraucht wird.
- [Wettbewerbsnachteil](wettbewerbsnachteil.md)
<br/>  Von Wartung überwältigte Teams können keine neuen Features liefern, was dazu führt, dass das Produkt hinter Wettbewerbern zurückfällt.

## Causes ▼

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Duplizierter Code vervielfacht die Wartungslast, da identische Korrekturen über alle Kopien hinweg angewendet werden müssen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte technische Schulden machen jede Wartungsaufgabe komplexer und zeitaufwendiger, als sie sein sollte.
- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener, unstrukturierter Code ist von Natur aus schwer zu warten und erfordert exzessiven Aufwand, um ihn sicher zu verstehen und zu modifizieren.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne Tests müssen Entwickler zusätzliche Zeit aufwenden, um manuell zu verifizieren, dass Wartungsänderungen bestehende Funktionalität nicht brechen.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Code, der schwer zu verstehen ist, erfordert unverhältnismäßig viel Zeit zur Wartung, da Entwickler ihn zunächst entziffern müssen, bevor sie Änderungen vornehmen.
- [API-Versionierungskonflikte](api-versionierungskonflikte.md)
<br/>  Das gleichzeitige Betreiben mehrerer inkompatibler API-Versionen vervielfacht die Codepfade, Tests und Dokumentation, die gleichzeitig gewartet werden müssen.

## Detection Methods ○
- **Zeiterfassung:** Nachverfolgung der Zeit, die das Team für Wartungsaufgaben versus neue Entwicklung aufwendet. Ein hohes Verhältnis ist ein klares Zeichen für ein Problem.
- **Fehlerdichte:** Messung der Anzahl der Fehler pro Codezeile. Eine hohe Fehlerdichte ist ein Zeichen dafür, dass die Codebasis schwer zu warten ist.
- **Code Churn:** Analyse der Historie der Codebasis, um zu sehen, welche Dateien am häufigsten modifiziert werden. Hoher Churn in bestimmten Dateien kann darauf hindeuten, dass sie eine Quelle hohen Wartungs-Overheads sind.
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrer Erfahrung mit Wartungsarbeit. Ihr Feedback kann eine wertvolle Informationsquelle sein.

## Examples
Ein Team ist für die Wartung einer großen, monolithischen Anwendung verantwortlich. Die Anwendung ist in einer alten Version einer Programmiersprache geschrieben und enthält viel duplizierten Code. Das Team verbringt die meiste Zeit damit, Fehler zu beheben und kleine Änderungen an der Anwendung vorzunehmen. Es bleibt sehr wenig Zeit für neue Feature-Entwicklung. Infolgedessen fällt die Anwendung hinter ihre Wettbewerber zurück, und das Geschäft beginnt, Marktanteile zu verlieren.
