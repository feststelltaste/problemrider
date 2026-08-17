---
title: Doppelter Aufwand
description: Mehrere Teammitglieder arbeiten unwissentlich an denselben Problemen
  oder implementieren unabhängig voneinander ähnliche Lösungen.
category:
- Communication
- Process
- Team
related_problems:
- slug: duplicated-work
  similarity: 0.95
- slug: duplicated-research-effort
  similarity: 0.85
- slug: team-confusion
  similarity: 0.7
- slug: communication-breakdown
  similarity: 0.7
- slug: code-duplication
  similarity: 0.65
- slug: team-coordination-issues
  similarity: 0.65
solutions:
- knowledge-sharing-practices
- cross-platform-frameworks
- knowledge-rotation
- knowledge-base
- clear-ownership-model
- team-boundaries-aligned-to-architecture
- documentation-as-code
- communities-of-practice
- master-data-stewardship
layout: problem
lang: de
en_slug: duplicated-effort
---

## Description

Doppelter Aufwand entsteht, wenn mehrere Teammitglieder unabhängig voneinander an denselben Problemen arbeiten, ähnliche Funktionalität implementieren oder dieselben Themen recherchieren, ohne zu bemerken, dass andere ähnliche Arbeit leisten. Dies stellt verschwendete Produktivität und verpasste Gelegenheiten für Zusammenarbeit, Wissensaustausch und effizientere Ressourcennutzung dar. Doppelter Aufwand deutet oft auf Kommunikationsprobleme oder unzureichende Koordinationsmechanismen innerhalb des Teams hin.

## Indicators ⟡

- Mehrere Teammitglieder entdecken, dass sie an ähnlichen Problemen gearbeitet haben
- Ähnlicher Code oder ähnliche Lösungen erscheinen in unterschiedlichen Teilen des Systems
- Teammitglieder recherchieren unabhängig voneinander dieselben Themen
- Arbeitszuweisungen überlappen sich ohne klare Koordination
- Unterschiedliche Teammitglieder kommen zu unterschiedlichen Schlussfolgerungen zu denselben technischen Fragen

## Symptoms ▲

- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Wenn mehrere Personen unwissentlich an demselben Problem arbeiten, stellt die redundante Arbeit direkt verschwendete Entwicklungsressourcen dar.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Unabhängige Implementierungen ähnlicher Funktionalität führen dazu, dass duplizierter Code in unterschiedlichen Teilen der Codebasis erscheint.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Der Team-Output sinkt, wenn mehrere Mitglieder Zeit damit verbringen, Probleme zu lösen, die nur einmal hätten gelöst werden müssen.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Unterschiedliche Entwickler, die dieselbe Funktionalität unabhängig voneinander implementieren, produzieren oft Lösungen mit subtil unterschiedlichem Verhalten.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Doppelter Aufwand verringert direkt die effektive Geschwindigkeit des Teams, da Kapazität für redundante Arbeit verbraucht wird.

## Causes ▼

- [Kommunikationszusammenbruch](kommunikationszusammenbruch.md)
<br/>  Schlechte Kommunikation verhindert, dass Teammitglieder wissen, woran andere arbeiten, was zu unkoordinierten parallelen Anstrengungen führt.
- [Team-Silos](team-silos.md)
<br/>  Wenn Teams oder Einzelpersonen isoliert arbeiten, fehlt ihnen Einblick in die Arbeit der anderen, was Duplizierung wahrscheinlich macht.
- [Probleme bei der Teamkoordination](probleme-bei-der-teamkoordination.md)
<br/>  Unzureichende Koordinationsmechanismen wie unklare Aufgabenzuweisungen und fehlende Arbeitsnachverfolgung ermöglichen überlappende Anstrengungen.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Unzureichende Sprint-Planung und Aufgabenaufteilung bedeuten, dass sich Arbeitszuweisungen überlappen, ohne dass jemand es bemerkt.
- [Teamverwirrung](teamverwirrung.md)
<br/>  Wenn Teammitglieder unklar über Verantwortlichkeiten sind und wer woran arbeitet, folgt doppelter Aufwand natürlicherweise.

## Detection Methods ○

- **Arbeitsüberlappungsanalyse:** Regelmäßige Überprüfung der Teamzuweisungen zur Identifikation potenzieller Überlappungen
- **Code-Ähnlichkeitserkennung:** Nutzung von Werkzeugen zur Identifikation ähnlicher Code-Implementierungen über die Codebasis hinweg
- **Recherchethemen-Tracking:** Beobachtung, was Teammitglieder recherchieren und untersuchen
- **Sprint-Planungs-Review:** Bewertung von Sprint-Plänen auf duplizierte oder überlappende Arbeit
- **Retrospektiven-Feedback:** Befragung von Teammitgliedern zu Fällen doppelten Aufwands, denen sie begegnet sind

## Examples

Zwei Entwickler verbringen jeweils eine Woche damit, Nutzerauthentifizierungs-Features für unterschiedliche Teile der Anwendung zu implementieren, ohne zu bemerken, dass sie einen gemeinsamen Authentifizierungsdienst nutzen könnten. Als sie ihre Lösungen während der Integration vergleichen, entdecken sie, dass sie dieselben Probleme unterschiedlich gelöst haben, was zusätzliche Arbeit erfordert, um einen konsistenten Ansatz zu schaffen. Ein weiteres Beispiel betrifft drei Teammitglieder, die unabhängig voneinander die Best Practices für die Implementierung von API-Rate-Limiting recherchieren, jeder verbringt mehrere Stunden mit dem Lesen von Dokumentation und dem Testen unterschiedlicher Ansätze. Sie kommen zu unterschiedlichen Schlussfolgerungen über die beste Lösung, und das Team muss zusätzliche Zeit aufwenden, um ihre Erkenntnisse abzugleichen und sich auf einen einzigen Ansatz zu einigen, was den benötigten Rechercheaufwand verdreifacht.
