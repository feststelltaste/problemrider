---
title: Einfache Sprache
description: Nutzung einfacher und klarer Formulierungen.
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/plain-language/
problems:
- user-confusion
- poor-user-experience-ux-design
- poor-documentation
- user-frustration
- difficult-developer-onboarding
- negative-user-feedback
- knowledge-gaps
- language-barriers
layout: solution
lang: de
en_slug: plain-language
related_solutions:
- slug: understandable-error-messages
  similarity: 0.8
- slug: consistent-terminology
  similarity: 0.8
- slug: intuitive-navigation
  similarity: 0.75
- slug: ubiquitous-language
  similarity: 0.75
- slug: visual-hierarchy
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.75
---

## Description

Einfache Sprache schreibt Oberflächentext, Fehlermeldungen und Dokumentation in klarer, alltäglicher Formulierung neu und ersetzt den Fachjargon und die kryptischen Codes, die sich in Legacy-Systemen ansammeln, während jede Entwicklergeneration Text hinzufügt, der für den nächsten Entwickler statt für den tatsächlichen Nutzer gedacht ist. Ein Fehler wie „Constraint violation: FK_CASE_PARTY_REF integrity check failed" bedeutet der Person, die ihn sieht, nichts und verwandelt sich zuverlässig in einen Support-Anruf, während derselbe Fehler, erklärt in Begriffen dessen, was passiert ist und was dagegen zu tun ist, jemandem erlaubt, ihn selbst zu lösen. Der Aufwand ist für ein weitläufiges Legacy-System mit Hunderten solcher Meldungen echt erheblich, und die Aufrechterhaltung des Standards erfordert anhaltende Wachsamkeit im Review, da Entwickler unkontrolliert zu der technischen Formulierung zurückkehren, die ihnen natürlich vorkommt.

## How to Apply ◆

> Legacy-Systeme sammeln technischen Jargon, kryptische Abkürzungen und entwicklerorientierte Sprache in ihren Oberflächen und ihrer Dokumentation an. Einfache Sprache ersetzt dies durch klares, nutzerorientiertes Schreiben.

- Prüfen Sie allen nutzerseitigen Text im Legacy-System, einschließlich Beschriftungen, Schaltflächen, Menüpunkten, Fehlermeldungen, Hilfetext und Tooltips. Markieren Sie Fälle von technischem Jargon, Abkürzungen und übermäßig komplexen Sätzen.
- Schreiben Sie Fehlermeldungen neu, um zu erklären, was passiert ist, warum es passiert ist und was der Nutzer dagegen tun kann. Ersetzen Sie Meldungen wie „ERR_4021: Transaction rollback" durch „Ihre Änderungen konnten nicht gespeichert werden, weil ein anderer Nutzer diesen Datensatz aktualisiert hat. Bitte aktualisieren Sie die Seite und versuchen Sie es erneut."
- Verwenden Sie handlungsorientierte Schaltflächenbeschriftungen, die beschreiben, was passieren wird, wie „Speichern und fortfahren" oder „Diesen Datensatz löschen", statt generischer Beschriftungen wie „OK", „Absenden" oder „Ausführen".
- Schreiben Sie kurze Sätze im Aktiv. Vermeiden Sie Passivkonstruktionen wie „Der Datensatz wurde aktualisiert", wenn „Wir haben den Datensatz aktualisiert" oder „Datensatz aktualisiert" klarer ist.
- Definieren Sie einen Schreibstil-Leitfaden für den gesamten nutzerseitigen Text, der Ton, Terminologie, Groß-/Kleinschreibung und Zeichensetzung abdeckt. Teilen Sie ihn mit allen Entwicklern, die Oberflächentext schreiben.
- Testen Sie neuen Text mit repräsentativen Nutzern, um Verständlichkeit zu verifizieren, besonders für kritische Arbeitsabläufe, bei denen Missverständnisse Konsequenzen haben.

## Tradeoffs ⇄

> Einfache Sprache macht das System zugänglicher und reduziert Fehler, erfordert aber Schreibfähigkeit und anhaltende Aufmerksamkeit für Sprachqualität.

**Vorteile:**

- Reduziert Nutzerverwirrung, indem Oberflächentext ohne Fachexpertise oder Systemerfahrung sofort verständlich gemacht wird.
- Verringert Support-Anfragen, die durch Nutzer verursacht werden, die nicht verstehen, was eine Schaltfläche tut oder was eine Fehlermeldung bedeutet.
- Verkürzt die Onboarding-Zeit, weil neue Nutzer die Oberfläche verstehen können, ohne Jargon oder Abkürzungen auswendig zu lernen.
- Macht das System einem breiteren Publikum zugänglich, einschließlich Nutzern mit geringerer Lesekompetenz oder Nicht-Muttersprachlern.

**Kosten und Risiken:**

- Das Neuschreiben allen nutzerseitigen Texts in einem großen Legacy-System ist ein erheblicher Aufwand, der neben anderen Verbesserungen priorisiert werden muss.
- Konsens über Alternativen in einfacher Sprache für etablierte Fachbegriffe zu erreichen kann schwierig sein, wenn verschiedene Gruppen starke Präferenzen haben.
- Die Übervereinfachung technischer Konzepte in einem expertenorientierten System kann für erfahrene Nutzer, die präzise Terminologie bevorzugen, herablassend wirken.
- Die Aufrechterhaltung von Standards für einfache Sprache erfordert Wachsamkeit während Code-Reviews, da Entwickler ohne Anleitung zu technischer Sprache zurückkehren.

## How It Could Be

> Legacy-Systeme fühlen sich oft nicht wegen mangelhafter Funktionalität feindselig gegenüber Nutzern an, sondern wegen undurchdringlicher Sprache.

Ein Legacy-Fallverwaltungssystem für Gerichte, das von Gerichtsschreibern genutzt wird, zeigt Fehlermeldungen an, die von Entwicklern geschrieben und nie auf Klarheit überprüft wurden. Meldungen wie „Constraint violation: FK_CASE_PARTY_REF integrity check failed" erscheinen, wenn ein Schreiber versucht, einen Parteidatensatz zu löschen, der noch mit einem aktiven Fall verknüpft ist. Die Schreiber haben keine Ahnung, was die Meldung bedeutet, und rufen entweder den IT-Support an oder probieren zufällige Dinge aus, bis der Fehler verschwindet. Das Team schreibt alle zweihundert Fehlermeldungen in einfacher Sprache neu, wobei die Constraint-Verletzungsmeldung zu „Diese Partei kann nicht entfernt werden, weil sie mit einem oder mehreren aktiven Fällen verknüpft ist. Um sie zu entfernen, weisen Sie zunächst die verknüpften Fälle neu zu oder schließen Sie sie" wird. Jede Meldung enthält nun eine spezifische Handlung, die der Nutzer ausführen kann. Support-Anrufe im Zusammenhang mit Fehlermeldungen sinken um mehr als die Hälfte, und Schreiber berichten, sich sicherer bei der Nutzung des Systems zu fühlen, weil sie Probleme unabhängig verstehen und darauf reagieren können.
