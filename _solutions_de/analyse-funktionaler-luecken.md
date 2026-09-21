---
title: Analyse funktionaler Lücken
description: Identifikation fehlender Funktionalität durch Vergleich von Fähigkeiten
  mit Anforderungen.
category:
- Requirements
- Business
problems:
- feature-gaps
- requirements-ambiguity
- inadequate-requirements-gathering
- modernization-roi-justification-failure
- stakeholder-frustration
- customer-dissatisfaction
- process-software-misfit
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: functional-gap-analysis
related_solutions:
- slug: requirements-analysis
  similarity: 0.75
- slug: risk-analysis
  similarity: 0.75
- slug: business-metrics
  similarity: 0.7
- slug: business-process-modeling
  similarity: 0.7
- slug: functional-debt-management
  similarity: 0.65
- slug: feature-driven-development
  similarity: 0.65
---

## Description

Die Analyse funktionaler Lücken ist ein strukturierter Vergleich zwischen dem, was ein System derzeit tut, und dem, was das Unternehmen tatsächlich braucht, dass es tut, und produziert eine explizite, priorisierte Liste von Diskrepanzen statt eines vagen Gefühls, dass das System „zurückfällt". Die Methode inventarisiert bestehende Fähigkeiten, sammelt aktuelle und antizipierte Anforderungen von Stakeholdern und klassifiziert die Unterschiede in fehlende Funktionalität, unterdurchschnittlich funktionierende Funktionalität und Funktionalität, die irrelevant geworden ist. Dies unterscheidet sich von rein technischen Bewertungen der Codequalität: Ein Legacy-System kann intern gut konstruiert sein und trotzdem das Unternehmen im Stich lassen, weil es für einen anderen Maßstab, Markt oder regulatorisches Umfeld gebaut wurde, als es jetzt operiert. In der Legacy-Modernisierungsarbeit liefert die Lückenanalyse die Evidenzbasis für die Entscheidung zwischen inkrementeller Erweiterung und gezieltem Ersatz, weil sie offenbart, ob Defizite oberflächlich und behebbar oder strukturell und durchgängig sind. Sie schützt auch vor zwei entgegengesetzten Versagensmodi: übermäßige Investition in die Modernisierung von Fähigkeiten, die das System bereits angemessen handhabt, und unzureichende Investition in die spezifischen Bereiche — oft neuere Geschäftsfelder oder Integrationsbedürfnisse —, in denen das Legacy-System nie darauf ausgelegt war zu konkurrieren.

## How to Apply ◆

- Dokumentieren Sie die aktuellen Fähigkeiten des Legacy-Systems systematisch: was es tut, wie gut es das tut und wo es zurückbleibt.
- Sammeln Sie aktuelle und künftige Geschäftsanforderungen von Stakeholdern und vergleichen Sie sie mit dem Fähigkeitsinventar des Legacy-Systems.
- Kategorisieren Sie Lücken: fehlende Features, unterdurchschnittliche Features, Features mit übermäßigen Workarounds und Features, die den Geschäftsbedürfnissen nicht mehr dienen.
- Priorisieren Sie Lücken nach geschäftlicher Auswirkung und strategischer Bedeutung, nicht nur danach, wie leicht sie zu schließen sind.
- Nutzen Sie die Ergebnisse der Lückenanalyse, um eine Modernisierungs-Roadmap mit klaren Meilensteinen und Erfolgskriterien aufzubauen.
- Überprüfen Sie die Lückenanalyse periodisch erneut, während sich Geschäftsanforderungen weiterentwickeln.

## Tradeoffs ⇄

**Vorteile:**
- Liefert ein klares, priorisiertes Bild davon, wo das Legacy-System die Geschäftsbedürfnisse nicht erfüllt.
- Schafft eine datengestützte Grundlage für Modernisierungsplanung und Investitionsrechtfertigung.
- Hilft, übermäßige Investition in Bereiche zu vermeiden, in denen das Legacy-System bereits angemessen ist.
- Richtet technische und geschäftliche Stakeholder auf ein gemeinsames Verständnis der Systemgrenzen aus.

**Kosten:**
- Die Durchführung einer gründlichen Lückenanalyse erfordert erhebliche Stakeholder-Beteiligung und Zeit.
- Die Anforderungserhebung kann widersprüchliche Bedürfnisse offenlegen, die schwer zu vereinbaren sind.
- Die Analyse stellt eine Momentaufnahme dar; Geschäftsanforderungen entwickeln sich weiter.
- Die Lückenanalyse allein löst keine Probleme; ihr muss Handlung folgen.

## How It Could Be

Ein Legacy-Lieferkettensystem bedient ein wachsendes Unternehmen, aber die Führung ist unsicher, ob in die Erweiterung des Legacy-Systems investiert oder es vollständig ersetzt werden soll. Das Team führt eine Analyse funktionaler Lücken durch, die die Fähigkeiten des Systems mit aktuellen Geschäftsanforderungen vergleicht, die von Lagerleitern, Beschaffungsteams und Logistikkoordinatoren erhoben wurden. Die Analyse offenbart, dass das System den inländischen Versand gut handhabt, aber vollständig die Unterstützung für internationale Logistik vermissen lässt (Zolldokumentation, Mehrwährungspreise, grenzüberschreitende Compliance), was den Hauptwachstumsbereich des Unternehmens darstellt. Dieser Befund macht die Modernisierungsentscheidung klar: Die Lücke ist zu groß, um sie mit inkrementellen Legacy-Erweiterungen zu schließen, und das Unternehmen fährt mit einem gezielten Ersatz der internationalen Logistikkomponenten fort, während das inländische Versandmodul beibehalten wird. Die Lückenanalyse spart Monate an Debatte und verhindert Investition in die Erweiterung von Legacy-Funktionalität, die bald ersetzt worden wäre.
