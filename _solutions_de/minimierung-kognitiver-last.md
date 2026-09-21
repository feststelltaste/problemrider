---
title: Minimierung kognitiver Last
description: Gestaltung der Nutzeroberfläche, damit sie intuitiv und leicht verständlich
  ist.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/cognitive-load-minimization/
problems:
- poor-user-experience-ux-design
- user-confusion
- user-frustration
- cognitive-overload
- increased-cognitive-load
- difficult-developer-onboarding
- negative-user-feedback
- shadow-systems
- mental-fatigue
- reduced-individual-productivity
- context-switching-overhead
layout: solution
lang: de
en_slug: cognitive-load-minimization
related_solutions:
- slug: intuitive-navigation
  similarity: 0.85
- slug: user-centered-design
  similarity: 0.8
- slug: consistent-user-interface
  similarity: 0.8
- slug: progressive-disclosure
  similarity: 0.8
- slug: visual-hierarchy
  similarity: 0.8
- slug: consistent-terminology
  similarity: 0.75
---

## Description

Minimierung kognitiver Last strukturiert eine Schnittstelle um das herum, was ein Nutzer tatsächlich zu tun versucht, statt um die interne Systemstruktur oder Datenbankstruktur, die der Legacy-Bildschirm zufällig direkt exponierte. Legacy-Schnittstellen zeigen routinemäßig jedes verfügbare Feld gleichzeitig, in welcher Reihenfolge auch immer das zugrunde liegende Schema vorschreibt, was Nutzer zwingt, Informationen visuell zu scannen und mental zu filtern, die ein neu designter Bildschirm ihnen für diese Aufgabe einfach nicht zeigen würde. Das Gruppieren verwandter Felder, das Verstecken selten benötigter Details hinter Progressive Disclosure und der Ersatz kryptischer interner Codes durch lesbare Beschriftungen verringert den mentalen Aufwand der Systemnutzung, muss aber getan werden, ohne Funktionalität zu vergraben, auf die Power-User, die das alte Layout auswendig gelernt haben, genuin weiterhin angewiesen sind.

## How to Apply ◆

> Legacy-Schnittstellen exponieren häufig interne Systemkomplexität direkt gegenüber Nutzern, was hohe kognitive Last schafft. Diese Last zu minimieren bedeutet, die Schnittstelle um Nutzeraufgaben statt um Systemstruktur herum zu restrukturieren.

- Auditieren Sie jeden Bildschirm auf Informationsdichte. Legacy-Systeme tendieren dazu, jedes verfügbare Datenfeld gleichzeitig anzuzeigen. Identifizieren Sie, welche Felder tatsächlich für jede Nutzeraufgabe benötigt werden, und verstecken Sie den Rest hinter Progressive Disclosure.
- Gruppieren Sie verwandte Steuerelemente und Informationen zusammen unter Nutzung visueller Nähe, Rahmen und Überschriften. Legacy-Formulare verstreuen häufig verwandte Felder über den Bildschirm in einer Reihenfolge, die das Datenbankschema statt das mentale Modell des Nutzers widerspiegelt.
- Nutzen Sie konsistente und vertraute Interaktionsmuster in der gesamten Anwendung. Wenn sich verschiedene Abschnitte eines Legacy-Systems für dieselbe Art von Aktion unterschiedlich verhalten, müssen Nutzer die Schnittstelle wiederholt neu lernen.
- Verringern Sie die Anzahl der zu einem beliebigen Zeitpunkt präsentierten Wahlmöglichkeiten. Legacy-Menüs mit Dutzenden von Optionen können in kategorisierte, durchsuchbare Befehlspaletten oder aufgabenorientierte Navigation umstrukturiert werden.
- Bieten Sie sinnvolle Standardwerte für Formularfelder basierend auf dem häufigsten Anwendungsfall, was die Anzahl der Entscheidungen verringert, die Nutzer für Routineaufgaben treffen müssen.
- Ersetzen Sie kryptische Codes und Abkürzungen, geerbt vom Legacy-System, durch menschenlesbare Beschriftungen. Viele Legacy-Systeme zeigen interne Kennungen, die für Endnutzer nichts bedeuten.

## Tradeoffs ⇄

> Die Verringerung kognitiver Last macht das System zugänglicher und effizienter, riskiert aber, Funktionalität zu verstecken, auf die Power-User angewiesen sind.

**Vorteile:**

- Verringert direkt Nutzerverwirrung und -frustration, indem nur die für die aktuelle Aufgabe relevanten Informationen und Optionen präsentiert werden.
- Verringert die Schulungszeit für neue Nutzer, weil eine einfachere Schnittstelle weniger Lernen erfordert.
- Verringert Fehlerraten, weil Nutzer weniger wahrscheinlich falsche Optionen wählen oder Daten in falsche Felder eingeben, wenn die Schnittstelle klar und fokussiert ist.
- Eliminiert die Motivation für Schattensysteme, indem das offizielle System für übliche Aufgaben genuin einfach zu nutzen gemacht wird.

**Kosten und Risiken:**

- Power-User, die das Legacy-Layout auswendig gelernt haben, könnten die vereinfachte Schnittstelle anfänglich langsamer finden, wenn ihre etablierten Workflows gestört werden.
- Das Verstecken selten genutzter Features hinter Progressive Disclosure erfordert sorgfältige Analyse, um zu vermeiden, Funktionalität zu vergraben, die manche Nutzergruppen regelmäßig brauchen.
- Die Neugestaltung der Informationsarchitektur in einem Legacy-System könnte Änderungen an Backend-APIs erfordern, wenn die aktuelle API-Struktur das alte UI-Layout spiegelt.
- Schrittweise Verbesserungen kognitiver Last können Inkonsistenz zwischen modernisierten und nicht modernisierten Abschnitten schaffen, was vorübergehend Verwirrung erhöht.

## How It Could Be

> Legacy-Systeme häufen über Jahrzehnte Schnittstellenkomplexität an, während Features ohne ganzheitliche Designüberlegung hinzugefügt werden.

Das Legacy-Sendungsverfolgungssystem eines Logistikunternehmens zeigt über sechzig Felder auf dem Hauptverfolgungsbildschirm, einschließlich interner Verarbeitungscodes, Datenbank-Zeitstempel und Systemflags, die nur für Entwickler bedeutsam sind. Disponenten verbringen erhebliche Zeit damit, den Bildschirm visuell zu scannen, um die fünf oder sechs Felder zu finden, die sie tatsächlich brauchen. Das Team führt kontextuelle Untersuchungen mit Disponenten durch und identifiziert drei primäre Aufgabenabläufe, jeder eine andere Teilmenge von Feldern erfordernd. Sie gestalten den Verfolgungsbildschirm als tabbed Interface neu, mit einer Zusammenfassungsansicht, die nur die kritischsten Sendungsinformationen zeigt, mit detaillierten Ansichten zugänglich über klar beschriftete Tabs. Disponenten berichten, dass ihr täglicher Workflow merklich schneller ist, und neue Disponenten erreichen Kompetenz in Tagen statt Wochen.
