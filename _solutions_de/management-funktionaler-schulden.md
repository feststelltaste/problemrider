---
title: Management funktionaler Schulden
description: Identifikation und Priorisierung problematischer Umsetzung funktionaler
  Anforderungen.
category:
- Management
- Requirements
problems:
- high-technical-debt
- feature-gaps
- accumulation-of-workarounds
- reduced-feature-quality
- delayed-bug-fixes
- customer-dissatisfaction
- declining-business-metrics
layout: solution
lang: de
en_slug: functional-debt-management
related_solutions:
- slug: technical-debt-backlog
  similarity: 0.8
- slug: debt-classification
  similarity: 0.75
- slug: debt-remediation-estimation
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
- slug: business-metrics
  similarity: 0.7
- slug: feature-driven-development
  similarity: 0.7
---

## Description

Das Management funktionaler Schulden behandelt Lücken und Defekte in dem, was ein System tut — im Gegensatz dazu, wie sauber es gebaut ist — als eigenständige, verfolgbare Kategorie technischer Schulden. Wo technische Schulden interne Codequalitätsprobleme wie Duplikation oder schlechte Struktur beschreiben, beschreiben funktionale Schulden nutzersichtbare Defizite: Features, die sich falsch oder unvollständig verhalten oder Workarounds erfordern, um überhaupt genutzt zu werden. In Legacy-Systemen sammeln sich funktionale Schulden still an, weil sie selten Build-Fehler oder Warnungen der statischen Analyse auslösen — sie treten nur durch Nutzerbeschwerden, Support-Tickets und informelle Erfahrungswerte über „den Export, der immer abschneidet" oder „den Bericht, den man zweimal laufen lassen muss" zutage. Sie zu managen bedeutet, ein explizites Inventar dieser Lücken aufzubauen, die Geschäftskosten jeder einzelnen zu bewerten statt ihr technisches Interesse, und diese Bewertung in die Priorisierung neben neuer Feature-Arbeit einfließen zu lassen. Diese Umdeutung ist für die Legacy-Modernisierung wichtig, weil funktionale Schulden oft das sind, was Stakeholder tatsächlich meinen, wenn sie sagen, ein System sei „veraltet", selbst wenn der zugrundeliegende Code technisch solide ist, und weil ein Modernisierungsvorhaben, das sie ignoriert, riskiert, dieselben funktionalen Defizite treu in einem neueren Technologie-Stack zu reproduzieren.

## How to Apply ◆

- Unterscheiden Sie funktionale Schulden (Features, die schlecht oder unvollständig funktionieren) von technischen Schulden (interne Codequalitätsprobleme) und verfolgen Sie sie separat.
- Inventarisieren Sie bekannte funktionale Lücken, Workarounds und teilweise implementierte Features im Legacy-System.
- Bewerten Sie die geschäftliche Auswirkung jedes funktionalen Schuldenpostens: wie viele Nutzer betroffen sind, welche Workarounds sie nutzen und welcher Geschäftswert verloren geht.
- Priorisieren Sie die Behebung funktionaler Schulden nach geschäftlicher Auswirkung, nicht nur nach technischer Leichtigkeit der Behebung.
- Reservieren Sie einen konsistenten Anteil der Entwicklungskapazität (z. B. 20 %) für die Behebung funktionaler Schulden neben neuer Feature-Entwicklung.
- Verfolgen Sie Trends funktionaler Schulden über die Zeit: Verbessert oder verschlechtert sich die funktionale Qualität des Legacy-Systems?

## Tradeoffs ⇄

**Vorteile:**
- Macht die Lücke zwischen dem, was das System tun sollte, und dem, was es tatsächlich tut, sichtbar und handhabbar.
- Priorisiert Behebungen nach geschäftlicher Auswirkung statt nach technischem Interesse.
- Verhindert, dass sich funktionale Schulden bis zu dem Punkt ansammeln, an dem das System unbrauchbar wird.
- Liefert Daten zur Rechtfertigung von Investitionen in die Legacy-System-Verbesserung.

**Kosten:**
- Die Katalogisierung funktionaler Schulden erfordert Input von Nutzern, Support-Teams und Entwicklern.
- Die Bewertung geschäftlicher Auswirkung kann subjektiv und politisch beeinflusst sein.
- Das Abwägen der Behebung funktionaler Schulden gegen neue Feature-Nachfrage erfordert laufende Verhandlung.
- Manche funktionalen Schulden können tief eingebettet und teuer zu beheben sein.

## How It Could Be

Ein Legacy-CRM-System hat über Jahre funktionale Schulden angesammelt: Suchergebnisse enthalten keine kürzlich hinzugefügten Kontakte, das Export-Feature schneidet große Datensätze still ab, und das Reporting-Modul berechnet Quartalssummen falsch, wenn Transaktionen Zeitzonen überspannen. Nutzer haben für jedes Problem Workarounds entwickelt, aber diese Workarounds verbrauchen wöchentlich Stunden an Mitarbeiterzeit. Das Team erstellt ein Register funktionaler Schulden, das die geschäftliche Auswirkung und Behebungskosten jedes Postens bewertet. Der Zeitzonenberechnungsfehler wird zuerst priorisiert, weil er die Genauigkeit der Finanzberichterstattung betrifft. Das Abschneide-Problem folgt an zweiter Stelle, weil es erhebliche Mitarbeiterzeit verschwendet. Über vier Quartale adressiert das Team systematisch die Posten mit der höchsten Auswirkung, und Nutzerzufriedenheitsumfragen zeigen deutliche Verbesserung, während langjährige Frustrationen endlich gelöst werden.
