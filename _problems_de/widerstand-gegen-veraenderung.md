---
title: Widerstand gegen Veränderung
description: Teams zögern, Teile des Systems zu refaktorieren oder zu verbessern,
  aufgrund wahrgenommenen Risikos und Aufwands, was zu Stagnation führt.
category:
- Code
- Process
- Team
related_problems:
- slug: fear-of-change
  similarity: 0.8
- slug: maintenance-paralysis
  similarity: 0.75
- slug: history-of-failed-changes
  similarity: 0.75
- slug: refactoring-avoidance
  similarity: 0.75
- slug: fear-of-breaking-changes
  similarity: 0.75
- slug: scope-change-resistance
  similarity: 0.7
solutions:
- blameless-postmortems
- architecture-workshops
- raising-user-awareness
- security-culture
- fair-source
- team-retrospectives
- pilot-projects
- psychological-safety-practices
- small-change-batches
- written-first-communication
- executive-sponsorship
layout: problem
lang: de
en_slug: resistance-to-change
---

## Description

Widerstand gegen Veränderung tritt auf, wenn Entwicklungsteams konsequent notwendige Verbesserungen, Refactoring oder Modernisierungsbemühungen vermeiden, aufgrund von Bedenken bezüglich Risiko, Aufwand oder Störung. Dieser Widerstand kann aus vergangenen negativen Erfahrungen entstehen, fehlendem Vertrauen in die Fähigkeit des Teams, Veränderung sicher zu managen, oder einer Organisationskultur, die davon abhält, Risiken einzugehen. Über die Zeit führt dieser Widerstand zu Systemstagnation und sich anhäufenden technischen Schulden.

## Indicators ⟡

- Verbesserungsinitiativen werden konsequent verschoben oder abgesagt
- Team-Diskussionen über Refactoring fokussieren sich primär auf Risiken statt Vorteile
- Workarounds werden gegenüber der Behebung zugrunde liegender Probleme bevorzugt
- Neue Anforderungen werden als Ergänzungen statt Verbesserungen an bestehendem Code implementiert
- Vorschläge für Systemverbesserungen erhalten skeptische oder negative Antworten

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Wenn Teams sich weigern, problematischen Code zu ändern, erstellen sie Workarounds statt Grundursachen zu beheben, was Komplexität hinzufügt.
- [Systemstagnation](systemstagnation.md)
<br/>  Anhaltender Widerstand gegen Verbesserungen verursacht, dass das System unverändert bleibt, während sich Geschäftsbedürfnisse und Technologie weiterentwickeln.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Das Vermeiden notwendigen Refactorings und notwendiger Verbesserungen erlaubt es technischen Schulden, sich ungebremst über die Zeit anzuhäufen.
- [Unfähigkeit zu innovieren](unfaehigkeit-zu-innovieren.md)
<br/>  Widerstand gegen die Änderung des bestehenden Systems verhindert die Übernahme neuer Ansätze, Technologien oder architektonischer Verbesserungen.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Unwilligkeit, die Codebasis zu verbessern, zwingt Entwickler, um bestehende Probleme herumzuarbeiten, was die Feature-Lieferung verlangsamt.

## Causes ▼

- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Berechtigte Angst, bestehende Funktionalität zu brechen, macht Teams zurückhaltend, funktionierenden Code anzufassen, selbst wenn er Verbesserung benötigt.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne Tests, die verifizieren, dass Änderungen bestehende Funktionalität nicht brechen, ist das wahrgenommene Risiko jeder Änderung hoch, was Verbesserungsbemühungen entmutigt.
- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Wenn Fehler bestraft werden, vermeiden Teammitglieder das Risiko, Änderungen vorzunehmen, die fehlschlagen und Schuldzuweisung nach sich ziehen könnten.
- [Wartungslähmung](wartungslaehmung.md)
<br/>  Teams, die nicht verifizieren können, dass Änderungen Funktionalität nicht brechen, werden gelähmt und widersetzen sich, irgendwelche Verbesserungen vorzunehmen.
- [Negative Erfahrungen aus der Vergangenheit](negative-erfahrungen-aus-der-vergangenheit.md)
<br/>  Vergangene negative Erfahrungen mit Änderungen (fehlgeschlagene Deployments, defekte Systeme) sind eine direkte Ursache dafür, dass Teams resistent werden.

## Detection Methods ○

- **Nachverfolgung von Verbesserungsvorschlägen:** Überwachung, wie viele Verbesserungsinitiativen begonnen versus abgeschlossen werden
- **Code-Alters-Analyse:** Identifikation von Codebereichen, die trotz bekannter Probleme nicht verbessert wurden
- **Team-Retrospektiven:** Diskussion von Einstellungen zu Veränderung und Verbesserungsbemühungen
- **Trendanalyse technischer Schulden:** Nachverfolgung, ob technische Schulden über die Zeit zu- oder abnehmen
- **Entscheidungsmusteranalyse:** Suche nach Mustern der Wahl von Workarounds über fundamentale Korrekturen

## Examples

Ein Entwicklungsteam identifiziert, dass sein Authentifizierungssystem Modernisierung benötigt, um neue Sicherheitsanforderungen zu unterstützen, aber jede Diskussion über die Aktualisierung endet mit Bedenken über das Brechen bestehender Integrationen. Statt das System zu modernisieren, fahren sie fort, patch-artige Sicherheitsmaßnahmen hinzuzufügen, die Komplexität erhöhen, während fundamentale Schwachstellen unangegangen bleiben. Ein weiteres Beispiel betrifft ein Team, das weiß, dass sein Datenbankdesign Performance-Probleme verursacht, sich aber gegen die Neugestaltung des Schemas wehrt, weil es Angst vor Datenmigrationsrisiken hat, und stattdessen zunehmend komplexe Caching-Schichten implementiert, die operativen Overhead hinzufügen, ohne die zugrunde liegenden Performance-Probleme zu lösen.
