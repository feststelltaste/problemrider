---
title: Annahmenbasierte Entwicklung
description: Entwickler treffen Entscheidungen auf Basis von Annahmen über Anforderungen
  oder Nutzerbedürfnisse, statt ihr Verständnis zu validieren.
category:
- Communication
- Process
- Requirements
related_problems:
- slug: decision-avoidance
  similarity: 0.6
- slug: feature-gaps
  similarity: 0.6
- slug: feedback-isolation
  similarity: 0.6
- slug: wasted-development-effort
  similarity: 0.6
- slug: requirements-ambiguity
  similarity: 0.55
- slug: work-blocking
  similarity: 0.55
solutions:
- boring-technologies
- design-by-contract
- technical-skills-development
- functional-spike
- prototypes
- prototyping
- technical-spike
- tracer-bullets
layout: problem
lang: de
en_slug: assumption-based-development
---

## Description

Annahmenbasierte Entwicklung entsteht, wenn Entwickler Entscheidungen über Funktionalität, Benutzeroberflächen-Design, Geschäftslogik oder technischen Ansatz auf Basis ihrer Annahmen darüber treffen, was Nutzer brauchen oder was Stakeholder wollen, statt diese Annahmen durch direkte Kommunikation oder Recherche zu validieren. Während manche Annahmen in der Entwicklung unvermeidlich sind, führt übermäßiges Vertrauen auf Annahmen zu Lösungen, die nicht den tatsächlichen Bedürfnissen entsprechen, und erfordert kostspielige Nacharbeit.

## Indicators ⟡

- Entwickler setzen die Umsetzung um, ohne klärende Fragen zu stellen
- Design-Entscheidungen werden ohne Rücksprache mit Nutzern oder Stakeholdern getroffen
- Geschäftslogik wird auf Basis der Interpretation von Anforderungen durch den Entwickler umgesetzt
- Grenzfälle und Fehlerbedingungen werden auf Basis von Entwickler-Annahmen behandelt
- Benutzeroberflächen-Designs werden ohne Nutzereingaben oder -tests erstellt

## Symptoms ▲

- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Wenn sich Annahmen über Anforderungen als falsch erweisen, muss abgeschlossene Arbeit nachbearbeitet werden, um den tatsächlichen Bedürfnissen zu entsprechen.
- [Fehlausgerichtete Liefergegenstände](fehlausgerichtete-liefergegenstaende.md)
<br/>  Software, die auf unvalidierten Annahmen aufgebaut ist, liefert Features, die nicht dem entsprechen, was Stakeholder tatsächlich brauchen.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Entwicklungsaufwand, der für den Bau von Features auf Basis falscher Annahmen aufgewendet wird, ist effektiv verschwendet.
- [Funktionslücken](funktionsluecken.md)
<br/>  Annahmen über Nutzerbedürfnisse führen dazu, dass die falschen Features gebaut werden, während tatsächliche Bedürfnisse unadressiert bleiben.

## Causes ▼

- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Wenn Anforderungen nicht ordentlich erhoben werden, füllen Entwickler Lücken mit ihren eigenen Annahmen.
- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Schlechte Kommunikationskanäle zwischen Stakeholdern und Entwicklern zwingen Entwickler dazu, Anforderungen zu erraten.
- [Termindruck](termindruck.md)
<br/>  Wenn ein knapper Zeitplan gerade in die Phase der Anforderungsklärung fällt, setzen Entwickler auf unvalidierten Annahmen fort, statt sich Zeit zu nehmen, das Verständnis mit Stakeholdern abzustimmen.
- [Feedback-Isolation](feedback-isolation.md)
<br/>  Wenn isolierte Teams sich entscheiden, weiterzuarbeiten, statt innezuhalten und zur Klärung zu eskalieren, füllen sie die entstehenden Informationslücken mit Annahmen über Nutzerbedürfnisse.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Wenn Dokumentation zu bestehenden Geschäftsregeln oder Systemverhalten fehlt oder unzuverlässig ist, raten Entwickler möglicherweise bei diesem spezifischen Verhalten, statt es nachzuschlagen; die meiste annahmenbasierte Entwicklung entsteht jedoch eher aus Lücken bei der Anforderungserhebung oder der Stakeholder-Kommunikation als aus allgemeiner Dokumentationsqualität.

## Detection Methods ○

- **Annahmen-Dokumentation:** Nachverfolgung, welche Annahmen Entwickler während der Entwicklung treffen
- **Häufigkeit der Stakeholder-Validierung:** Beobachtung, wie oft Entwickler Annahmen mit Stakeholdern abgleichen
- **Nacharbeitsanalyse:** Analyse, wie viel Nacharbeit durch falsche Annahmen verursacht wird
- **Korrelation von Nutzerfeedback:** Vergleich von Nutzerfeedback mit ursprünglichen Entwickler-Annahmen
- **Bewertung der Anforderungsklarheit:** Bewertung, wie klar und spezifisch Anforderungen vor Beginn der Entwicklung sind

## Examples

Ein Entwicklungsteam baut ein Kundensuche-Feature unter der Annahme, dass Nutzer hauptsächlich nach Firmennamen suchen werden, und optimiert die Suchoberfläche und -algorithmen entsprechend. Nach dem Deployment des Features stellt es fest, dass Nutzer tatsächlich am häufigsten nach Ansprechpartnernamen und E-Mail-Adressen suchen, was die Suchoberfläche frustrierend in der Nutzung macht und zu Suchergebnissen schlechter Qualität führt. Das Suche-Feature muss neu entworfen und neu gebaut werden, um die tatsächlichen Suchmuster zu unterstützen. Ein weiteres Beispiel betrifft Entwickler, die ein Reporting-System bauen und annehmen, dass Nutzer Daten in Echtzeit sehen wollen, weshalb sie komplexe Echtzeit-Datenverarbeitung umsetzen. Tatsächlich bevorzugen die Nutzer jedoch, mit stabilen täglichen Datenschnappschüssen zu arbeiten, um Konsistenz in ihrer Analyse zu gewährleisten, und die Echtzeit-Updates erschweren die Erstellung reproduzierbarer Berichte. Die Echtzeit-Komplexität schafft keinen Mehrwert und erzeugt Wartungsaufwand für Funktionalität, die Nutzer nicht wollen.
